use omniscient::config::Config;
use omniscient::embed::MockEmbedder;
use omniscient::engine::Engine;
use std::fs;
use tempfile::tempdir;

#[tokio::test]
async fn end_to_end_semantic_search_with_mock() {
    let repo = tempdir().unwrap();
    fs::write(repo.path().join("auth.rs"),
        "pub fn renew_credentials() -> Token {\n    refresh_token()\n}\n").unwrap();
    fs::write(repo.path().join("util.py"), "def add(a, b):\n    return a + b\n").unwrap();

    let cfg = Config::default_for(repo.path().to_path_buf());
    let engine = Engine::new_with_embedder(cfg, Box::new(MockEmbedder::new("mock-v1", 64)))
        .await.unwrap();

    let entries = engine.search("renew_credentials", Some(3)).await.unwrap();
    assert!(entries.iter().any(|e| e.path == "auth.rs"));
}
