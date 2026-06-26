//! Embeddings: Embedder trait + llama.cpp HTTP backend (/v1/embeddings) + mock.
use crate::config::EmbedderConfig;
use crate::error::{Error, Result};
use async_trait::async_trait;

#[async_trait]
pub trait Embedder: Send + Sync {
    fn id(&self) -> &str;
    fn dim(&self) -> usize;
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;
}

pub fn l2_normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 { for x in v.iter_mut() { *x /= norm; } }
}

pub async fn build_embedder(cfg: &EmbedderConfig) -> Result<Box<dyn Embedder>> {
    Ok(Box::new(LlamaCppEmbedder::connect(cfg.base_url.clone(), cfg.model.clone()).await?))
}

// ---- Deterministic test embedder ----
pub struct MockEmbedder { id: String, dim: usize }
impl MockEmbedder {
    pub fn new(id: &str, dim: usize) -> Self { Self { id: id.into(), dim } }
}
#[async_trait]
impl Embedder for MockEmbedder {
    fn id(&self) -> &str { &self.id }
    fn dim(&self) -> usize { self.dim }
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|t| {
            let mut v = vec![0f32; self.dim];
            for (i, b) in t.bytes().enumerate() {
                v[i % self.dim] += (b as f32 + 1.0) * ((i % 7 + 1) as f32);
            }
            l2_normalize(&mut v);
            v
        }).collect())
    }
}

// ---- llama.cpp HTTP backend ----
pub struct LlamaCppEmbedder {
    base_url: String,
    model: String,
    dim: usize,
    client: reqwest::Client,
}

impl LlamaCppEmbedder {
    pub async fn connect(base_url: String, model: String) -> Result<Self> {
        let mut e = Self { base_url, model, dim: 0, client: reqwest::Client::new() };
        let probe = e.embed_raw(&["probe".to_string()]).await?;
        e.dim = probe.first().map(|r| r.len()).unwrap_or(0);
        if e.dim == 0 {
            return Err(Error::Embed("embeddings endpoint returned an empty vector (dim 0)".into()));
        }
        Ok(e)
    }

    async fn embed_raw(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        #[derive(serde::Serialize)]
        struct Req<'a> { model: &'a str, input: &'a [String] }
        #[derive(serde::Deserialize)]
        struct Item { embedding: Vec<f32> }
        #[derive(serde::Deserialize)]
        struct Resp { data: Vec<Item> }
        let url = format!("{}/v1/embeddings", self.base_url.trim_end_matches('/'));
        let resp = self.client.post(&url)
            .json(&Req { model: &self.model, input: texts })
            .send().await
            .map_err(|e| Error::Embed(format!("POST {url} failed: {e}. Is llama.cpp serving the embedding model?")))?
            .error_for_status().map_err(|e| Error::Embed(e.to_string()))?
            .json::<Resp>().await.map_err(|e| Error::Embed(e.to_string()))?;
        Ok(resp.data.into_iter().map(|it| it.embedding).collect())
    }
}

#[async_trait]
impl Embedder for LlamaCppEmbedder {
    fn id(&self) -> &str { &self.model }
    fn dim(&self) -> usize { self.dim }
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() { return Ok(vec![]); }
        let mut rows = self.embed_raw(texts).await?;
        for r in &mut rows { l2_normalize(r); }
        Ok(rows)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn mock_embeds_are_normalized_and_stable() {
        let e = MockEmbedder::new("mock-v1", 16);
        let a = e.embed(&["hello".into()]).await.unwrap();
        let b = e.embed(&["hello".into()]).await.unwrap();
        assert_eq!(a, b);
        assert_eq!(a[0].len(), 16);
        let norm: f32 = a[0].iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    #[tokio::test]
    async fn distinct_texts_differ() {
        let e = MockEmbedder::new("mock-v1", 16);
        let v = e.embed(&["alpha".into(), "beta".into()]).await.unwrap();
        assert_ne!(v[0], v[1]);
    }

    #[test]
    fn id_and_dim() {
        let e = MockEmbedder::new("mock-v1", 16);
        assert_eq!(e.id(), "mock-v1");
        assert_eq!(e.dim(), 16);
    }
}

#[cfg(test)]
mod live {
    use super::*;
    #[tokio::test]
    #[ignore = "requires a running llama.cpp embeddings server"]
    async fn live_embed_dim_and_norm() {
        let e = LlamaCppEmbedder::connect("http://localhost:8080".into(), "qwen3-embedding-4b".into())
            .await.unwrap();
        assert!(e.dim() > 0);
        let v = e.embed(&["fn main() {}".into()]).await.unwrap();
        let n: f32 = v[0].iter().map(|x| x*x).sum::<f32>().sqrt();
        assert!((n - 1.0).abs() < 1e-4);
    }
}
