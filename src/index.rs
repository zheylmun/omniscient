//! Vector index over LanceDB, guarded by embedder id/dim.
use crate::error::{Error, Result};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use arrow_array::{
    Array, FixedSizeListArray, Float32Array, RecordBatch,
    StringArray, UInt32Array, types::Float32Type,
};
use arrow_schema::{DataType, Field, Schema};
use futures::TryStreamExt;
use lancedb::query::{ExecutableQuery, QueryBase, Select};
use lancedb::{Connection, DistanceType, Table};

#[derive(Debug, Clone)]
pub struct StoredChunk {
    pub path: String,
    pub start_line: usize,
    pub end_line: usize,
    pub language: String,
    pub symbol: Option<String>,
    pub text: String,
    pub file_hash: String,
    pub vector: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct Hit { pub chunk: StoredChunk, pub score: f32 }

#[derive(serde::Serialize, serde::Deserialize)]
struct Meta {
    embedder_id: String,
    dim: usize,
    // Defaults to 0 for indexes written before chunker versioning existed, so
    // they mismatch the current CHUNKER_VERSION (>= 1) and rebuild once.
    #[serde(default)]
    chunker_version: u32,
}

pub struct Index {
    dim: usize,
    table: Table,
    rebuilt: bool,
}

fn schema_for(dim: usize) -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("path", DataType::Utf8, false),
        Field::new("start_line", DataType::UInt32, false),
        Field::new("end_line", DataType::UInt32, false),
        Field::new("language", DataType::Utf8, false),
        Field::new("symbol", DataType::Utf8, true),
        Field::new("text", DataType::Utf8, false),
        Field::new("file_hash", DataType::Utf8, false),
        Field::new("vector",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), dim as i32),
            false),
    ]))
}

impl Index {
    pub async fn open(dir: &Path, embedder_id: &str, dim: usize, chunker_version: u32) -> Result<Index> {
        std::fs::create_dir_all(dir)?;
        let meta_path = dir.join("meta.json");
        let existing: Option<Meta> = std::fs::read_to_string(&meta_path).ok()
            .and_then(|s| serde_json::from_str(&s).ok());
        let mismatch = existing.as_ref()
            .map(|m| m.embedder_id != embedder_id || m.dim != dim || m.chunker_version != chunker_version)
            .unwrap_or(false);

        let conn: Connection = lancedb::connect(dir.join("lance").to_string_lossy().as_ref())
            .execute().await.map_err(|e| Error::Index(e.to_string()))?;

        let has_table = |names: &[String]| names.iter().any(|t| t == "chunks");
        let names = conn.table_names().execute().await.map_err(|e| Error::Index(e.to_string()))?;

        let mut rebuilt = false;
        if mismatch && has_table(&names) {
            conn.drop_table("chunks", &[]).await.map_err(|e| Error::Index(e.to_string()))?;
            rebuilt = true;
        }

        let names = conn.table_names().execute().await.map_err(|e| Error::Index(e.to_string()))?;
        let table = if has_table(&names) {
            conn.open_table("chunks").execute().await.map_err(|e| Error::Index(e.to_string()))?
        } else {
            conn.create_empty_table("chunks", schema_for(dim)).execute().await
                .map_err(|e| Error::Index(e.to_string()))?
        };

        std::fs::write(&meta_path,
            serde_json::to_string(&Meta { embedder_id: embedder_id.into(), dim, chunker_version }).unwrap())?;

        Ok(Index { dim, table, rebuilt })
    }

    pub fn rebuilt(&self) -> bool { self.rebuilt }

    pub async fn delete_file(&self, path: &str) -> Result<()> {
        self.table.delete(&format!("path = '{}'", path.replace('\'', "''")))
            .await.map_err(|e| Error::Index(e.to_string()))?;
        Ok(())
    }

    pub async fn upsert_file(&self, path: &str, chunks: Vec<StoredChunk>) -> Result<()> {
        self.delete_file(path).await?;
        if chunks.is_empty() { return Ok(()); }
        let schema = schema_for(self.dim);
        let batch = build_batch(&schema, &chunks, self.dim)?;
        self.table.add(vec![batch]).execute().await.map_err(|e| Error::Index(e.to_string()))?;
        Ok(())
    }

    pub async fn file_hashes(&self) -> Result<HashMap<String, String>> {
        let batches: Vec<RecordBatch> = self.table.query()
            .select(Select::columns(&["path", "file_hash"]))
            .execute().await.map_err(|e| Error::Index(e.to_string()))?
            .try_collect().await.map_err(|e| Error::Index(e.to_string()))?;
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
        self.table.count_rows(None).await.map_err(|e| Error::Index(e.to_string()))
    }

    pub async fn search(&self, query_vec: &[f32], k: usize) -> Result<Vec<Hit>> {
        let batches: Vec<RecordBatch> = self.table.query()
            .nearest_to(query_vec).map_err(|e| Error::Index(e.to_string()))?
            .distance_type(DistanceType::Cosine)
            .limit(k)
            .execute().await.map_err(|e| Error::Index(e.to_string()))?
            .try_collect().await.map_err(|e| Error::Index(e.to_string()))?;

        let mut hits = Vec::new();
        for b in &batches {
            let paths = str_col(b, "path")?;
            let langs = str_col(b, "language")?;
            let texts = str_col(b, "text")?;
            let hashes = str_col(b, "file_hash")?;
            let syms = str_col(b, "symbol")?;
            let starts = u32_col(b, "start_line")?;
            let ends = u32_col(b, "end_line")?;
            let dist = b.column_by_name("_distance")
                .and_then(|c| c.as_any().downcast_ref::<Float32Array>())
                .ok_or_else(|| Error::Index("_distance column missing".into()))?;
            for i in 0..b.num_rows() {
                let symbol = if syms.is_null(i) { None } else { Some(syms.value(i).to_string()) };
                hits.push(Hit {
                    score: 1.0 - dist.value(i),
                    chunk: StoredChunk {
                        path: paths.value(i).to_string(),
                        start_line: starts.value(i) as usize,
                        end_line: ends.value(i) as usize,
                        language: langs.value(i).to_string(),
                        symbol,
                        text: texts.value(i).to_string(),
                        file_hash: hashes.value(i).to_string(),
                        vector: vec![],
                    },
                });
            }
        }
        hits.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        Ok(hits)
    }
}

fn str_col<'a>(b: &'a RecordBatch, name: &str) -> Result<&'a StringArray> {
    b.column_by_name(name).and_then(|c| c.as_any().downcast_ref::<StringArray>())
        .ok_or_else(|| Error::Index(format!("column {name} missing or not Utf8")))
}
fn u32_col<'a>(b: &'a RecordBatch, name: &str) -> Result<&'a UInt32Array> {
    b.column_by_name(name).and_then(|c| c.as_any().downcast_ref::<UInt32Array>())
        .ok_or_else(|| Error::Index(format!("column {name} missing or not UInt32")))
}

fn build_batch(schema: &Arc<Schema>, chunks: &[StoredChunk], dim: usize) -> Result<RecordBatch> {
    let paths = StringArray::from(chunks.iter().map(|c| c.path.clone()).collect::<Vec<_>>());
    let starts = UInt32Array::from(chunks.iter().map(|c| c.start_line as u32).collect::<Vec<_>>());
    let ends = UInt32Array::from(chunks.iter().map(|c| c.end_line as u32).collect::<Vec<_>>());
    let langs = StringArray::from(chunks.iter().map(|c| c.language.clone()).collect::<Vec<_>>());
    let syms = StringArray::from(chunks.iter().map(|c| c.symbol.clone()).collect::<Vec<Option<String>>>());
    let texts = StringArray::from(chunks.iter().map(|c| c.text.clone()).collect::<Vec<_>>());
    let hashes = StringArray::from(chunks.iter().map(|c| c.file_hash.clone()).collect::<Vec<_>>());
    let vectors = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
        chunks.iter().map(|c| Some(c.vector.iter().map(|&v| Some(v)).collect::<Vec<_>>())),
        dim as i32,
    );
    RecordBatch::try_new(schema.clone(), vec![
        Arc::new(paths), Arc::new(starts), Arc::new(ends), Arc::new(langs),
        Arc::new(syms), Arc::new(texts), Arc::new(hashes), Arc::new(vectors),
    ]).map_err(|e| Error::Index(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn chunk(path: &str, hash: &str, line: usize, vec: Vec<f32>) -> StoredChunk {
        StoredChunk {
            path: path.into(), start_line: line, end_line: line + 1,
            language: "rust".into(), symbol: Some("f".into()),
            text: format!("code at {line}"), file_hash: hash.into(), vector: vec,
        }
    }

    #[tokio::test]
    async fn upsert_search_roundtrip() {
        let dir = tempdir().unwrap();
        let idx = Index::open(dir.path(), "mock-v1", 3, 1).await.unwrap();
        idx.upsert_file("a.rs", vec![
            chunk("a.rs","h1",1,vec![1.0,0.0,0.0]),
            chunk("a.rs","h1",5,vec![0.0,1.0,0.0]),
        ]).await.unwrap();
        let hits = idx.search(&[1.0,0.0,0.0], 1).await.unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].chunk.start_line, 1);
    }

    #[tokio::test]
    async fn upsert_replaces_old_rows_for_file() {
        let dir = tempdir().unwrap();
        let idx = Index::open(dir.path(), "mock-v1", 3, 1).await.unwrap();
        idx.upsert_file("a.rs", vec![chunk("a.rs","h1",1,vec![1.0,0.0,0.0])]).await.unwrap();
        idx.upsert_file("a.rs", vec![chunk("a.rs","h2",9,vec![1.0,0.0,0.0])]).await.unwrap();
        let hashes = idx.file_hashes().await.unwrap();
        assert_eq!(hashes.get("a.rs"), Some(&"h2".to_string()));
        let hits = idx.search(&[1.0,0.0,0.0], 5).await.unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].chunk.start_line, 9);
    }

    #[tokio::test]
    async fn model_id_mismatch_triggers_rebuild() {
        let dir = tempdir().unwrap();
        {
            let idx = Index::open(dir.path(), "mock-v1", 3, 1).await.unwrap();
            idx.upsert_file("a.rs", vec![chunk("a.rs","h1",1,vec![1.0,0.0,0.0])]).await.unwrap();
        }
        let idx = Index::open(dir.path(), "different-model", 3, 1).await.unwrap();
        assert!(idx.rebuilt());
        assert!(idx.file_hashes().await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn chunker_version_mismatch_triggers_rebuild() {
        let dir = tempdir().unwrap();
        {
            let idx = Index::open(dir.path(), "mock-v1", 3, 1).await.unwrap();
            idx.upsert_file("a.rs", vec![chunk("a.rs","h1",1,vec![1.0,0.0,0.0])]).await.unwrap();
        }
        // Same embedder, bumped chunker version: stale chunks must be dropped.
        let idx = Index::open(dir.path(), "mock-v1", 3, 2).await.unwrap();
        assert!(idx.rebuilt());
        assert!(idx.file_hashes().await.unwrap().is_empty());
    }
}
