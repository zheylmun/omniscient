//! Vector index over `LanceDB`, guarded by embedder id/dim.
use crate::error::{Error, Result};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use arrow_array::{
    Array, FixedSizeListArray, Float32Array, RecordBatch, RecordBatchIterator, StringArray,
    UInt32Array, types::Float32Type,
};
use arrow_schema::{DataType, Field, Schema};
use futures::TryStreamExt;
use lancedb::query::{ExecutableQuery, QueryBase};
use lancedb::{Connection, DistanceType, Table};

#[derive(Debug, Clone)]
pub struct StoredChunk {
    pub path: String,
    pub start_line: usize,
    pub end_line: usize,
    /// Position within the file's chunk list. Disambiguates the merge key, and
    /// orders sub-line pieces that necessarily share a line number.
    pub chunk_index: usize,
    pub language: String,
    pub symbol: Option<String>,
    pub text: String,
    pub file_hash: String,
    pub vector: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct Hit {
    pub chunk: StoredChunk,
    pub score: f32,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct Meta {
    embedder_id: String,
    dim: usize,
    // Defaults to 0 for indexes written before chunker versioning existed, so
    // they mismatch the current CHUNKER_VERSION (>= 1) and rebuild once.
    #[serde(default)]
    chunker_version: u32,
}

#[derive(Clone)]
pub struct Index {
    dim: usize,
    table: Table,
    meta: Table,
    rebuilt: bool,
}

fn schema_for(dim: usize) -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("path", DataType::Utf8, false),
        Field::new("start_line", DataType::UInt32, false),
        Field::new("end_line", DataType::UInt32, false),
        // Per-file ordinal (0-based, tree-walk order). Part of the merge key so two
        // chunks that share a physical line range (e.g. `type A=u8; type B=u16;` or
        // minified TS) stay distinct — line ranges alone are not unique.
        Field::new("chunk_index", DataType::UInt32, false),
        Field::new("language", DataType::Utf8, false),
        Field::new("symbol", DataType::Utf8, true),
        Field::new("text", DataType::Utf8, false),
        Field::new("file_hash", DataType::Utf8, false),
        Field::new(
            "vector",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                i32::try_from(dim).expect("embedding dimension fits in i32"),
            ),
            false,
        ),
    ]))
}

fn meta_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("path", DataType::Utf8, false),
        Field::new("file_hash", DataType::Utf8, false),
    ]))
}

/// SQL predicate matching one file's rows, with single quotes escaped.
fn path_eq_filter(path: &str) -> String {
    format!("path = '{}'", path.replace('\'', "''"))
}

/// One-time upgrade migration: an index built before `file_meta` existed has populated
/// `chunks` but empty meta. Rebuild meta from the distinct `(path, file_hash)` pairs in
/// the chunks table so `file_hashes()` doesn't report every file as new.
async fn backfill_meta_from_chunks(table: &Table, meta: &Table) -> Result<()> {
    use lancedb::query::Select;
    let batches: Vec<RecordBatch> = table
        .query()
        .select(Select::columns(&["path", "file_hash"]))
        .execute()
        .await
        .map_err(|e| Error::Index(e.to_string()))?
        .try_collect()
        .await
        .map_err(|e| Error::Index(e.to_string()))?;
    let mut map: HashMap<String, String> = HashMap::new();
    for b in &batches {
        let paths = str_col(b, "path")?;
        let hashes = str_col(b, "file_hash")?;
        for i in 0..b.num_rows() {
            map.insert(paths.value(i).to_string(), hashes.value(i).to_string());
        }
    }
    if map.is_empty() {
        return Ok(());
    }
    let (paths, hashes): (Vec<String>, Vec<String>) = map.into_iter().unzip();
    let schema = meta_schema();
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(StringArray::from(paths)),
            Arc::new(StringArray::from(hashes)),
        ],
    )
    .map_err(|e| Error::Index(e.to_string()))?;
    meta.add(vec![batch])
        .execute()
        .await
        .map_err(|e| Error::Index(e.to_string()))?;
    Ok(())
}

impl Index {
    pub async fn open(
        dir: &Path,
        embedder_id: &str,
        dim: usize,
        chunker_version: u32,
    ) -> Result<Index> {
        std::fs::create_dir_all(dir)?;
        let meta_path = dir.join("meta.json");
        let existing: Option<Meta> = std::fs::read_to_string(&meta_path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok());
        // Unknown provenance means rebuild. A table we cannot attribute to an
        // embedder id, dim and chunker version may predate the current schema,
        // and reusing it fails every merge_insert — a fatal error that wedges
        // reconcile permanently. An unnecessary rebuild is the cheap direction.
        let mismatch = existing.as_ref().is_none_or(|m| {
            m.embedder_id != embedder_id || m.dim != dim || m.chunker_version != chunker_version
        });

        // Zero read-consistency interval: re-resolve the on-disk manifest before
        // every read. Without this, a long-lived handle (e.g. the `serve` process)
        // keeps pointing at fragment files that a separate `reindex` has deleted,
        // failing every query with "Object ... not found" until the process is
        // restarted. See `handle_survives_external_index_rebuild`.
        let conn: Connection = lancedb::connect(dir.join("lance").to_string_lossy().as_ref())
            .read_consistency_interval(std::time::Duration::ZERO)
            .execute()
            .await
            .map_err(|e| Error::Index(e.to_string()))?;

        let has_table = |names: &[String], t: &str| names.iter().any(|n| n == t);
        let names = conn
            .table_names()
            .execute()
            .await
            .map_err(|e| Error::Index(e.to_string()))?;

        let mut rebuilt = false;
        if mismatch && has_table(&names, "chunks") {
            conn.drop_table("chunks", &[])
                .await
                .map_err(|e| Error::Index(e.to_string()))?;
            if has_table(&names, "file_meta") {
                conn.drop_table("file_meta", &[])
                    .await
                    .map_err(|e| Error::Index(e.to_string()))?;
            }
            rebuilt = true;
        }

        let names = conn
            .table_names()
            .execute()
            .await
            .map_err(|e| Error::Index(e.to_string()))?;
        let chunks_existed = has_table(&names, "chunks");
        let meta_existed = has_table(&names, "file_meta");
        let table = if chunks_existed {
            conn.open_table("chunks")
                .execute()
                .await
                .map_err(|e| Error::Index(e.to_string()))?
        } else {
            conn.create_empty_table("chunks", schema_for(dim))
                .execute()
                .await
                .map_err(|e| Error::Index(e.to_string()))?
        };

        // Lightweight metadata table: one row per file (path, file_hash).
        // file_hashes() reads from this instead of scanning every chunk row,
        // so it scales O(files) not O(chunks).
        let meta = if meta_existed {
            conn.open_table("file_meta")
                .execute()
                .await
                .map_err(|e| Error::Index(e.to_string()))?
        } else {
            conn.create_empty_table("file_meta", meta_schema())
                .execute()
                .await
                .map_err(|e| Error::Index(e.to_string()))?
        };

        // Upgrade migration: an index that predates the file_meta table has chunks but
        // no meta table, so backfill it (once) to avoid re-embedding every file on the
        // first reconcile. Gating on table existence — not row counts — keeps the happy
        // path (already-migrated or brand-new index) free of full-table count scans.
        if !rebuilt && chunks_existed && !meta_existed {
            backfill_meta_from_chunks(&table, &meta).await?;
        }

        std::fs::write(
            &meta_path,
            serde_json::to_string(&Meta {
                embedder_id: embedder_id.into(),
                dim,
                chunker_version,
            })
            .unwrap(),
        )?;

        Ok(Index {
            dim,
            table,
            meta,
            rebuilt,
        })
    }

    pub fn rebuilt(&self) -> bool {
        self.rebuilt
    }

    pub async fn delete_file(&self, path: &str) -> Result<()> {
        let filter = path_eq_filter(path);
        self.table
            .delete(&filter)
            .await
            .map_err(|e| Error::Index(e.to_string()))?;
        self.meta
            .delete(&filter)
            .await
            .map_err(|e| Error::Index(e.to_string()))?;
        Ok(())
    }

    /// Idempotently record `(path, file_hash)` in the lightweight metadata table via a
    /// `merge_insert` on `path` (update if present, insert if new). Private: the meta
    /// table is an internal invariant of `upsert_file`/`delete_file`.
    async fn write_file_meta(&self, path: &str, file_hash: &str) -> Result<()> {
        let schema = meta_schema();
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(vec![path.to_string()])),
                Arc::new(StringArray::from(vec![file_hash.to_string()])),
            ],
        )
        .map_err(|e| Error::Index(e.to_string()))?;
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
        let mut op = self.meta.merge_insert(&["path"]);
        op.when_matched_update_all(None);
        op.when_not_matched_insert_all();
        op.execute(Box::new(reader))
            .await
            .map_err(|e| Error::Index(e.to_string()))?;
        Ok(())
    }

    /// Replace all rows for `path` with `chunks` in a single atomic `merge_insert`
    /// keyed on `(path, chunk_index)`: matching chunks are updated, new ones inserted,
    /// and stale chunks for this path deleted. The key uses the per-file ordinal rather
    /// than the line range because two top-level defs can share a physical line (e.g.
    /// minified TS, or `type A=u8; type B=u16;`), which would collide as merge keys and
    /// abort the whole operation. A concurrent `search` therefore sees all-old or all-new,
    /// never a gap or a superset. The operation is idempotent, so an interrupted reconcile
    /// that re-upserts the same content self-heals instead of duplicating rows. `file_meta`
    /// is updated in the same call and kept in lockstep.
    ///
    /// `file_hash` is passed explicitly rather than read off `chunks[0]` because a
    /// file that is present but yields no chunks still has one, and still needs it
    /// recorded — see the empty-chunks branch below.
    pub async fn upsert_file(
        &self,
        path: &str,
        file_hash: &str,
        chunks: Vec<StoredChunk>,
    ) -> Result<()> {
        if chunks.is_empty() {
            // The file exists but produced no chunks (empty, or whitespace only).
            // Drop any rows it left behind, but still record its hash: without a
            // `file_meta` row `diff` reports it as changed on every reconcile
            // forever, repeating this delete on every search that reconciles.
            self.table
                .delete(&path_eq_filter(path))
                .await
                .map_err(|e| Error::Index(e.to_string()))?;
            return self.write_file_meta(path, file_hash).await;
        }
        let new_hash = file_hash.to_string();
        let schema = schema_for(self.dim);
        let batch = build_batch(&schema, &chunks, self.dim)?;
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
        // Single atomic commit: update matching chunks, insert new ones, and delete
        // this file's stale chunks (scoped by path) that the new set no longer covers.
        // Idempotent — re-running with identical content converges to one set of rows.
        let mut op = self.table.merge_insert(&["path", "chunk_index"]);
        op.when_matched_update_all(None);
        op.when_not_matched_insert_all();
        op.when_not_matched_by_source_delete(Some(path_eq_filter(path)));
        op.execute(Box::new(reader))
            .await
            .map_err(|e| Error::Index(e.to_string()))?;
        self.write_file_meta(path, &new_hash).await?;
        Ok(())
    }

    pub async fn file_hashes(&self) -> Result<HashMap<String, String>> {
        // Read from the lightweight meta table (one row per file) instead of
        // scanning every chunk row — O(files) not O(chunks).
        let batches: Vec<RecordBatch> = self
            .meta
            .query()
            .execute()
            .await
            .map_err(|e| Error::Index(e.to_string()))?
            .try_collect()
            .await
            .map_err(|e| Error::Index(e.to_string()))?;
        let mut map = HashMap::new();
        for b in &batches {
            let paths = str_col(b, "path")?;
            let hashes = str_col(b, "file_hash")?;
            for i in 0..b.num_rows() {
                map.insert(paths.value(i).to_string(), hashes.value(i).to_string());
            }
        }
        Ok(map)
    }

    pub async fn chunk_count(&self) -> Result<usize> {
        self.table
            .count_rows(None)
            .await
            .map_err(|e| Error::Index(e.to_string()))
    }

    pub async fn search(&self, query_vec: &[f32], k: usize) -> Result<Vec<Hit>> {
        let batches: Vec<RecordBatch> = self
            .table
            .query()
            .nearest_to(query_vec)
            .map_err(|e| Error::Index(e.to_string()))?
            .distance_type(DistanceType::Cosine)
            .limit(k)
            .execute()
            .await
            .map_err(|e| Error::Index(e.to_string()))?
            .try_collect()
            .await
            .map_err(|e| Error::Index(e.to_string()))?;

        let mut hits = Vec::new();
        for b in &batches {
            let paths = str_col(b, "path")?;
            let langs = str_col(b, "language")?;
            let texts = str_col(b, "text")?;
            let hashes = str_col(b, "file_hash")?;
            let syms = str_col(b, "symbol")?;
            let starts = u32_col(b, "start_line")?;
            let ends = u32_col(b, "end_line")?;
            let idxs = u32_col(b, "chunk_index")?;
            let dist = b
                .column_by_name("_distance")
                .and_then(|c| c.as_any().downcast_ref::<Float32Array>())
                .ok_or_else(|| Error::Index("_distance column missing".into()))?;
            for i in 0..b.num_rows() {
                let symbol = if syms.is_null(i) {
                    None
                } else {
                    Some(syms.value(i).to_string())
                };
                hits.push(Hit {
                    score: 1.0 - dist.value(i),
                    chunk: StoredChunk {
                        path: paths.value(i).to_string(),
                        start_line: starts.value(i) as usize,
                        end_line: ends.value(i) as usize,
                        chunk_index: idxs.value(i) as usize,
                        language: langs.value(i).to_string(),
                        symbol,
                        text: texts.value(i).to_string(),
                        file_hash: hashes.value(i).to_string(),
                        vector: vec![],
                    },
                });
            }
        }
        hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(hits)
    }
}

fn str_col<'a>(b: &'a RecordBatch, name: &str) -> Result<&'a StringArray> {
    b.column_by_name(name)
        .and_then(|c| c.as_any().downcast_ref::<StringArray>())
        .ok_or_else(|| Error::Index(format!("column {name} missing or not Utf8")))
}
fn u32_col<'a>(b: &'a RecordBatch, name: &str) -> Result<&'a UInt32Array> {
    b.column_by_name(name)
        .and_then(|c| c.as_any().downcast_ref::<UInt32Array>())
        .ok_or_else(|| Error::Index(format!("column {name} missing or not UInt32")))
}

fn build_batch(schema: &Arc<Schema>, chunks: &[StoredChunk], dim: usize) -> Result<RecordBatch> {
    let paths = StringArray::from(chunks.iter().map(|c| c.path.clone()).collect::<Vec<_>>());
    // Line numbers are display metadata; clamp rather than panic on a pathological
    // file with more than u32::MAX lines.
    let starts = UInt32Array::from(
        chunks
            .iter()
            .map(|c| u32::try_from(c.start_line).unwrap_or(u32::MAX))
            .collect::<Vec<_>>(),
    );
    let ends = UInt32Array::from(
        chunks
            .iter()
            .map(|c| u32::try_from(c.end_line).unwrap_or(u32::MAX))
            .collect::<Vec<_>>(),
    );
    // Position within the file's chunk list — the disambiguator in the merge key.
    let indices = UInt32Array::from(
        chunks
            .iter()
            .map(|c| u32::try_from(c.chunk_index).unwrap_or(u32::MAX))
            .collect::<Vec<_>>(),
    );
    let langs = StringArray::from(
        chunks
            .iter()
            .map(|c| c.language.clone())
            .collect::<Vec<_>>(),
    );
    let syms = StringArray::from(
        chunks
            .iter()
            .map(|c| c.symbol.clone())
            .collect::<Vec<Option<String>>>(),
    );
    let texts = StringArray::from(chunks.iter().map(|c| c.text.clone()).collect::<Vec<_>>());
    let hashes = StringArray::from(
        chunks
            .iter()
            .map(|c| c.file_hash.clone())
            .collect::<Vec<_>>(),
    );
    let vectors = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
        chunks
            .iter()
            .map(|c| Some(c.vector.iter().map(|&v| Some(v)).collect::<Vec<_>>())),
        i32::try_from(dim).expect("embedding dimension fits in i32"),
    );
    RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(paths),
            Arc::new(starts),
            Arc::new(ends),
            Arc::new(indices),
            Arc::new(langs),
            Arc::new(syms),
            Arc::new(texts),
            Arc::new(hashes),
            Arc::new(vectors),
        ],
    )
    .map_err(|e| Error::Index(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn chunk(path: &str, hash: &str, index: usize, line: usize, vec: Vec<f32>) -> StoredChunk {
        StoredChunk {
            path: path.into(),
            start_line: line,
            end_line: line + 1,
            chunk_index: index,
            language: "rust".into(),
            symbol: Some("f".into()),
            text: format!("code at {line}"),
            file_hash: hash.into(),
            vector: vec,
        }
    }

    #[tokio::test]
    async fn upsert_search_roundtrip() {
        let dir = tempdir().unwrap();
        let idx = Index::open(dir.path(), "mock-v1", 3, 1).await.unwrap();
        idx.upsert_file(
            "a.rs",
            "h1",
            vec![
                chunk("a.rs", "h1", 0, 1, vec![1.0, 0.0, 0.0]),
                chunk("a.rs", "h1", 1, 5, vec![0.0, 1.0, 0.0]),
            ],
        )
        .await
        .unwrap();
        let hits = idx.search(&[1.0, 0.0, 0.0], 1).await.unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].chunk.start_line, 1);
    }

    #[tokio::test]
    async fn upsert_replaces_old_rows_for_file() {
        let dir = tempdir().unwrap();
        let idx = Index::open(dir.path(), "mock-v1", 3, 1).await.unwrap();
        idx.upsert_file(
            "a.rs",
            "h1",
            vec![chunk("a.rs", "h1", 0, 1, vec![1.0, 0.0, 0.0])],
        )
        .await
        .unwrap();
        idx.upsert_file(
            "a.rs",
            "h2",
            vec![chunk("a.rs", "h2", 0, 9, vec![1.0, 0.0, 0.0])],
        )
        .await
        .unwrap();
        let hashes = idx.file_hashes().await.unwrap();
        assert_eq!(hashes.get("a.rs"), Some(&"h2".to_string()));
        let hits = idx.search(&[1.0, 0.0, 0.0], 5).await.unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].chunk.start_line, 9);
    }

    #[tokio::test]
    async fn upsert_replaces_with_new_chunk_shape_and_keeps_new_rows() {
        // Re-chunking a changed file can produce a different number of chunks.
        // add-then-delete-by-hash must end with exactly the new rows: no stale
        // old-hash rows, and none of the just-added new rows wrongly deleted.
        let dir = tempdir().unwrap();
        let idx = Index::open(dir.path(), "mock-v1", 3, 1).await.unwrap();
        idx.upsert_file(
            "a.rs",
            "h1",
            vec![chunk("a.rs", "h1", 0, 1, vec![1.0, 0.0, 0.0])],
        )
        .await
        .unwrap();
        idx.upsert_file(
            "a.rs",
            "h2",
            vec![
                chunk("a.rs", "h2", 0, 10, vec![1.0, 0.0, 0.0]),
                chunk("a.rs", "h2", 1, 20, vec![0.0, 1.0, 0.0]),
            ],
        )
        .await
        .unwrap();

        assert_eq!(
            idx.chunk_count().await.unwrap(),
            2,
            "exactly the new rows remain"
        );
        assert_eq!(
            idx.file_hashes().await.unwrap().get("a.rs"),
            Some(&"h2".to_string())
        );
        let hits = idx.search(&[0.0, 1.0, 0.0], 5).await.unwrap();
        assert!(
            hits.iter().any(|h| h.chunk.start_line == 20),
            "new rows are queryable"
        );
    }

    #[tokio::test]
    async fn upsert_same_content_twice_is_idempotent() {
        let dir = tempfile::tempdir().unwrap();
        let idx = Index::open(dir.path(), "mock-v1", 3, 1).await.unwrap();
        let rows = vec![
            chunk("a.rs", "h1", 0, 1, vec![1.0, 0.0, 0.0]),
            chunk("a.rs", "h1", 1, 5, vec![0.0, 1.0, 0.0]),
        ];
        idx.upsert_file("a.rs", "h1", rows.clone()).await.unwrap();
        // Re-upsert the exact same content (the interrupted-reconcile / stale-meta case).
        idx.upsert_file("a.rs", "h1", rows).await.unwrap();
        assert_eq!(
            idx.chunk_count().await.unwrap(),
            2,
            "re-upserting identical content must not duplicate rows"
        );
        let hashes = idx.file_hashes().await.unwrap();
        assert_eq!(hashes.get("a.rs"), Some(&"h1".to_string()));
    }

    #[tokio::test]
    async fn upsert_handles_two_chunks_on_the_same_line() {
        // Two top-level defs can share a physical line (minified TS, `type A=u8; type B=u16;`),
        // yielding chunks with identical (start_line, end_line). Keying merge_insert on the
        // line range would make those collide and abort the whole upsert (leaving the file
        // unindexed and reconcile stuck); the chunk_index key must keep them distinct.
        let dir = tempdir().unwrap();
        let idx = Index::open(dir.path(), "mock-v1", 3, 1).await.unwrap();
        let mut a = chunk("a.rs", "h1", 0, 5, vec![1.0, 0.0, 0.0]);
        let mut b = chunk("a.rs", "h1", 1, 5, vec![0.0, 1.0, 0.0]);
        a.symbol = Some("A".into());
        b.symbol = Some("B".into());
        assert_eq!(
            (a.start_line, a.end_line),
            (b.start_line, b.end_line),
            "test premise: both chunks share the same line range"
        );
        idx.upsert_file("a.rs", "h1", vec![a, b]).await.unwrap();
        assert_eq!(
            idx.chunk_count().await.unwrap(),
            2,
            "both same-line chunks are stored, not collapsed"
        );
        // A second upsert of identical content must still converge (idempotent).
        let mut a2 = chunk("a.rs", "h1", 0, 5, vec![1.0, 0.0, 0.0]);
        let mut b2 = chunk("a.rs", "h1", 1, 5, vec![0.0, 1.0, 0.0]);
        a2.symbol = Some("A".into());
        b2.symbol = Some("B".into());
        idx.upsert_file("a.rs", "h1", vec![a2, b2]).await.unwrap();
        assert_eq!(
            idx.chunk_count().await.unwrap(),
            2,
            "re-upsert of same-line chunks must not duplicate"
        );
        let hits = idx.search(&[0.0, 1.0, 0.0], 5).await.unwrap();
        assert!(
            hits.iter().any(|h| h.chunk.symbol.as_deref() == Some("B")),
            "the second same-line chunk is queryable"
        );
    }

    #[tokio::test]
    async fn upsert_empty_chunks_removes_file() {
        let dir = tempdir().unwrap();
        let idx = Index::open(dir.path(), "mock-v1", 3, 1).await.unwrap();
        idx.upsert_file(
            "a.rs",
            "h1",
            vec![chunk("a.rs", "h1", 0, 1, vec![1.0, 0.0, 0.0])],
        )
        .await
        .unwrap();
        idx.upsert_file("a.rs", "h2", vec![]).await.unwrap();
        assert_eq!(
            idx.chunk_count().await.unwrap(),
            0,
            "empty upsert drops the file's rows"
        );
    }

    #[tokio::test]
    async fn handle_survives_external_index_rebuild() {
        // Reproduces the footgun where a long-lived server (e.g. the MCP `serve`
        // process) holds an Index handle while a *separate* process runs `reindex`,
        // which wipes `.omniscient/` and rebuilds from scratch. The stale handle
        // must not keep pointing at deleted fragment files.
        let dir = tempdir().unwrap();
        let idx = Index::open(dir.path(), "mock-v1", 3, 1).await.unwrap();
        idx.upsert_file(
            "a.rs",
            "h1",
            vec![chunk("a.rs", "h1", 0, 1, vec![1.0, 0.0, 0.0])],
        )
        .await
        .unwrap();

        // Simulate `reindex`: blow away the dataset dir and rebuild it via a fresh
        // handle with different contents.
        std::fs::remove_dir_all(dir.path().join("lance")).unwrap();
        {
            let rebuilt = Index::open(dir.path(), "mock-v1", 3, 1).await.unwrap();
            rebuilt
                .upsert_file(
                    "b.rs",
                    "h2",
                    vec![chunk("b.rs", "h2", 0, 7, vec![1.0, 0.0, 0.0])],
                )
                .await
                .unwrap();
        }

        // The original handle must reload to the rebuilt dataset, not error out on
        // the now-deleted fragment files.
        let hits = idx.search(&[1.0, 0.0, 0.0], 5).await.unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].chunk.path, "b.rs");
    }

    #[tokio::test]
    async fn model_id_mismatch_triggers_rebuild() {
        let dir = tempdir().unwrap();
        {
            let idx = Index::open(dir.path(), "mock-v1", 3, 1).await.unwrap();
            idx.upsert_file(
                "a.rs",
                "h1",
                vec![chunk("a.rs", "h1", 0, 1, vec![1.0, 0.0, 0.0])],
            )
            .await
            .unwrap();
        }
        let idx = Index::open(dir.path(), "different-model", 3, 1)
            .await
            .unwrap();
        assert!(idx.rebuilt());
        assert!(idx.file_hashes().await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn open_backfills_meta_from_chunks_when_missing() {
        let dir = tempfile::tempdir().unwrap();
        {
            let idx = Index::open(dir.path(), "mock-v1", 3, 1).await.unwrap();
            idx.upsert_file(
                "a.rs",
                "h1",
                vec![chunk("a.rs", "h1", 0, 1, vec![1.0, 0.0, 0.0])],
            )
            .await
            .unwrap();
        }
        // Simulate an index written before file_meta existed: drop the meta table.
        let conn = lancedb::connect(dir.path().join("lance").to_string_lossy().as_ref())
            .execute()
            .await
            .unwrap();
        conn.drop_table("file_meta", &[]).await.unwrap();
        // Reopen: backfill must repopulate meta so the file isn't seen as new.
        let idx = Index::open(dir.path(), "mock-v1", 3, 1).await.unwrap();
        let hashes = idx.file_hashes().await.unwrap();
        assert_eq!(
            hashes.get("a.rs"),
            Some(&"h1".to_string()),
            "meta must be backfilled from the chunks table on open"
        );
        assert_eq!(
            idx.chunk_count().await.unwrap(),
            1,
            "backfill must not touch chunks"
        );
    }

    #[tokio::test]
    async fn chunker_version_mismatch_triggers_rebuild() {
        let dir = tempdir().unwrap();
        {
            let idx = Index::open(dir.path(), "mock-v1", 3, 1).await.unwrap();
            idx.upsert_file(
                "a.rs",
                "h1",
                vec![chunk("a.rs", "h1", 0, 1, vec![1.0, 0.0, 0.0])],
            )
            .await
            .unwrap();
        }
        // Same embedder, bumped chunker version: stale chunks must be dropped.
        let idx = Index::open(dir.path(), "mock-v1", 3, 2).await.unwrap();
        assert!(idx.rebuilt());
        assert!(idx.file_hashes().await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn corrupt_meta_forces_a_rebuild() {
        // A truncated meta.json must not let a stale-schema table survive: the
        // first merge_insert would fail on the missing column, and index write
        // errors are fatal, so every reconcile — and every search — would fail
        // permanently until .omniscient/ was deleted by hand.
        let dir = tempdir().unwrap();
        let idx = Index::open(dir.path(), "embedder-a", 8, 4).await.unwrap();
        idx.upsert_file("a.rs", "h", vec![chunk("a.rs", "h", 0, 1, vec![0.0; 8])])
            .await
            .unwrap();
        assert_eq!(idx.chunk_count().await.unwrap(), 1);
        drop(idx);

        std::fs::write(dir.path().join("meta.json"), "{ truncated").unwrap();

        let idx = Index::open(dir.path(), "embedder-a", 8, 4).await.unwrap();
        assert_eq!(
            idx.chunk_count().await.unwrap(),
            0,
            "unreadable meta must rebuild, not reuse a table of unknown provenance"
        );
    }

    #[tokio::test]
    async fn missing_meta_forces_a_rebuild() {
        let dir = tempdir().unwrap();
        let idx = Index::open(dir.path(), "embedder-a", 8, 4).await.unwrap();
        idx.upsert_file("a.rs", "h", vec![chunk("a.rs", "h", 0, 1, vec![0.0; 8])])
            .await
            .unwrap();
        drop(idx);

        std::fs::remove_file(dir.path().join("meta.json")).unwrap();

        let idx = Index::open(dir.path(), "embedder-a", 8, 4).await.unwrap();
        assert_eq!(idx.chunk_count().await.unwrap(), 0);
    }
}
