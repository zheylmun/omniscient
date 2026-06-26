//! Engine: ties freshness + embed + index + distill into the always-fresh search path.
use crate::chunk::chunk_file;
use crate::config::Config;
use crate::distill::{distill_context, ContextEntry};
use crate::embed::{build_embedder, Embedder};
use crate::error::{Error, Result};
use crate::freshness::{diff, scan};
use crate::index::{Index, StoredChunk};
use std::path::Path;

const MAX_WINDOW_LINES: usize = 80;

pub struct Engine {
    config: Config,
    embedder: Box<dyn Embedder>,
    index: Index,
}

impl Engine {
    pub async fn new(config: Config) -> Result<Engine> {
        let embedder = build_embedder(&config.embedder).await?;
        Self::new_with_embedder(config, embedder).await
    }

    pub async fn new_with_embedder(config: Config, embedder: Box<dyn Embedder>) -> Result<Engine> {
        let dir = config.repo_root.join(".omniscient");
        let index = Index::open(&dir, embedder.id(), embedder.dim().max(1)).await?;
        Ok(Engine { config, embedder, index })
    }

    pub fn embedder_id(&self) -> &str { self.embedder.id() }

    pub async fn stats(&self) -> Result<(usize, usize)> {
        Ok((self.index.file_hashes().await?.len(), self.index.chunk_count().await?))
    }

    pub async fn refresh(&self) -> Result<()> {
        let current = scan(&self.config.repo_root)?;
        let stored = self.index.file_hashes().await?;
        let delta = diff(&current, &stored);
        let hash_of: std::collections::HashMap<&str, &str> =
            current.iter().map(|s| (s.path.as_str(), s.hash.as_str())).collect();

        for path in &delta.changed {
            let abs = self.config.repo_root.join(path);
            let source = match std::fs::read_to_string(&abs) { Ok(s) => s, Err(_) => continue };
            let chunks = chunk_file(Path::new(path), &source, MAX_WINDOW_LINES)?;
            if chunks.is_empty() { continue; }
            let texts: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();
            let vectors = self.embedder.embed(&texts).await?;
            if vectors.len() != texts.len() {
                return Err(Error::Embed(format!(
                    "embedder returned {} vectors for {} inputs (file {path})",
                    vectors.len(), texts.len()
                )));
            }
            let file_hash = hash_of.get(path.as_str()).copied().unwrap_or("").to_string();
            let stored_chunks: Vec<StoredChunk> = chunks.into_iter().zip(vectors).map(|(c, v)| StoredChunk {
                path: path.clone(), start_line: c.start_line, end_line: c.end_line,
                language: c.language, symbol: c.symbol, text: c.text,
                file_hash: file_hash.clone(), vector: v,
            }).collect();
            self.index.upsert_file(path, stored_chunks).await?;
        }
        for path in &delta.deleted { self.index.delete_file(path).await?; }
        Ok(())
    }

    /// Embed a single string, enforcing the embedder contract that exactly one
    /// vector comes back (so a misbehaving endpoint errors instead of panicking).
    async fn embed_one(&self, text: &str) -> Result<Vec<f32>> {
        let mut vs = self.embedder.embed(&[text.to_string()]).await?;
        if vs.len() != 1 {
            return Err(Error::Embed(format!(
                "embedder returned {} vectors for 1 input", vs.len()
            )));
        }
        Ok(vs.remove(0))
    }

    pub async fn search(&self, query: &str, k: Option<usize>) -> Result<Vec<ContextEntry>> {
        self.refresh().await?;
        let k = k.unwrap_or(self.config.search.default_k);
        let qv = self.embed_one(query).await?;
        let hits = self.index.search(&qv, k).await?;
        Ok(distill_context(hits, self.config.strip_comments, self.config.search.token_budget))
    }

    pub async fn read_file(&self, path: &str, focus: Option<&str>) -> Result<Vec<ContextEntry>> {
        let abs = self.config.repo_root.join(path);
        let source = std::fs::read_to_string(&abs)?;
        let chunks = chunk_file(Path::new(path), &source, MAX_WINDOW_LINES)?;
        match focus {
            None => Ok(chunks.into_iter().map(|c| ContextEntry {
                path: path.to_string(), start_line: c.start_line, end_line: c.end_line,
                language: c.language, symbol: c.symbol,
                code: c.text.lines().next().unwrap_or("").to_string(),
                score: 0.0,
                why_matched: "outline".into(),
            }).collect()),
            Some(f) => {
                if chunks.is_empty() { return Ok(vec![]); }
                let texts: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();
                let fv = self.embed_one(f).await?;
                let cvs = self.embedder.embed(&texts).await?;
                if cvs.len() != texts.len() {
                    return Err(Error::Embed(format!(
                        "embedder returned {} vectors for {} inputs", cvs.len(), texts.len()
                    )));
                }
                let mut scored: Vec<(f32, &crate::chunk::Chunk)> =
                    chunks.iter().enumerate().map(|(i, c)| (dot(&fv, &cvs[i]), c)).collect();
                scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
                Ok(scored.into_iter().take(5).map(|(score, c)| ContextEntry {
                    path: path.to_string(), start_line: c.start_line, end_line: c.end_line,
                    language: c.language.clone(), symbol: c.symbol.clone(),
                    code: c.text.clone(), score, why_matched: format!("focus similarity {score:.3}"),
                }).collect())
            }
        }
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 { a.iter().zip(b).map(|(x, y)| x * y).sum() }

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::embed::MockEmbedder;
    use async_trait::async_trait;
    use std::fs;
    use tempfile::tempdir;

    async fn engine_for(root: std::path::PathBuf) -> Engine {
        let cfg = Config::default_for(root);
        Engine::new_with_embedder(cfg, Box::new(MockEmbedder::new("mock-v1", 64))).await.unwrap()
    }

    /// Misbehaving embedder that returns no vectors — exercises the embed_one guard.
    struct ZeroEmbedder;
    #[async_trait]
    impl Embedder for ZeroEmbedder {
        fn id(&self) -> &str { "zero" }
        fn dim(&self) -> usize { 64 }
        async fn embed(&self, _texts: &[String]) -> Result<Vec<Vec<f32>>> { Ok(vec![]) }
    }

    #[tokio::test]
    async fn search_errors_when_embedder_returns_no_vectors() {
        // Empty repo: refresh embeds nothing, so this isolates embed_one(query).
        let repo = tempdir().unwrap();
        let cfg = Config::default_for(repo.path().to_path_buf());
        let engine = Engine::new_with_embedder(cfg, Box::new(ZeroEmbedder)).await.unwrap();
        let err = engine.search("anything", Some(3)).await.unwrap_err();
        assert!(matches!(err, Error::Embed(_)), "expected Error::Embed, got {err:?}");
    }

    #[tokio::test]
    async fn indexes_and_finds_relevant_file() {
        let repo = tempdir().unwrap();
        fs::write(repo.path().join("auth.rs"),
            "pub fn renew_credentials() -> Token {\n    refresh_token()\n}\n").unwrap();
        fs::write(repo.path().join("math.rs"),
            "pub fn add(a: i32, b: i32) -> i32 { a + b }\n").unwrap();
        let engine = engine_for(repo.path().to_path_buf()).await;
        let entries = engine.search("renew_credentials", Some(3)).await.unwrap();
        assert!(entries.iter().any(|e| e.path == "auth.rs"));
    }

    #[tokio::test]
    async fn refresh_picks_up_new_and_deleted_files() {
        let repo = tempdir().unwrap();
        fs::write(repo.path().join("a.rs"), "fn a(){}\n").unwrap();
        let engine = engine_for(repo.path().to_path_buf()).await;
        engine.refresh().await.unwrap();
        fs::write(repo.path().join("b.rs"), "fn b(){}\n").unwrap();
        fs::remove_file(repo.path().join("a.rs")).unwrap();
        let entries = engine.search("b", Some(5)).await.unwrap();
        assert!(entries.iter().any(|e| e.path == "b.rs"));
        assert!(!entries.iter().any(|e| e.path == "a.rs"));
    }

    #[tokio::test]
    async fn read_file_outline_and_focus() {
        let repo = tempdir().unwrap();
        fs::write(repo.path().join("lib.rs"),
            "pub fn alpha() {}\npub fn beta() {}\n").unwrap();
        let engine = engine_for(repo.path().to_path_buf()).await;

        // Outline (no focus): one entry per chunk, why_matched == "outline".
        let outline = engine.read_file("lib.rs", None).await.unwrap();
        assert!(!outline.is_empty());
        assert!(outline.iter().all(|e| e.why_matched == "outline"));
        assert!(outline.iter().any(|e| e.symbol.as_deref() == Some("alpha")));

        // Focus: ranked by similarity, why_matched mentions the focus.
        let focus = engine.read_file("lib.rs", Some("alpha")).await.unwrap();
        assert!(!focus.is_empty());
        assert!(focus.iter().all(|e| e.why_matched.contains("focus similarity")));
    }

    #[tokio::test]
    async fn index_persists_across_engine_reopen() {
        let repo = tempdir().unwrap();
        fs::write(repo.path().join("a.rs"), "pub fn a() {}\n").unwrap();
        let (files, chunks) = {
            let engine = engine_for(repo.path().to_path_buf()).await;
            engine.refresh().await.unwrap();
            engine.stats().await.unwrap()
        };
        assert!(files >= 1 && chunks >= 1);

        // Reopen a fresh Engine over the same repo dir (same embedder id) — the
        // LanceDB index must persist, and re-refreshing unchanged files must not
        // duplicate rows.
        let engine2 = engine_for(repo.path().to_path_buf()).await;
        assert_eq!(engine2.stats().await.unwrap(), (files, chunks), "index persisted");
        engine2.refresh().await.unwrap();
        assert_eq!(engine2.stats().await.unwrap(), (files, chunks), "unchanged files not re-indexed");
    }
}
