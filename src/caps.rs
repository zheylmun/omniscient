//! Embedding-server capability discovery.
//!
//! Three probes, because llama.cpp has two deployment shapes and only one probe
//! reports slots. A bare `llama-server` answers `GET /props` about the model it
//! loaded. `llama serve` is a supervisor: it spawns a child process per model on
//! an ephemeral port, and its *unscoped* `/props` describes the supervisor
//! itself, reporting `n_ctx: 0` and no `total_slots` at all. Scoping the request
//! with `?model=<id>` makes it proxy through to the backend, which is why
//! `embed::probe_caps` tries that first; plain `/props` second, for a server that
//! rejects the parameter; and `GET /v1/models` last, whose `data[].meta.n_ctx`
//! is sometimes the only window on offer but never carries a slot count.
//!
//! Both parsers are failure-soft by design: `meta` is an optional field on an
//! otherwise-standard `OpenAI` route, so a non-llama.cpp endpoint yields `None`
//! rather than an error and the caller falls back to configuration.

/// Which probe produced these caps. Reported by `diagnostics` so a router that
/// hides its backend's real context window is visible rather than inferred.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum CapsSource {
    /// Neither probe was informative; the configured fallback applies.
    #[default]
    None,
    /// `GET /props` — a bare `llama-server`.
    Props,
    /// `GET /v1/models` — behind a `llama serve` supervisor.
    Models,
}

/// What the embedding endpoint told us about its own limits. Every field is
/// optional: a non-llama.cpp `OpenAI`-compatible router will 404 `/props`, and we
/// must degrade to configured defaults rather than fail.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ServerCaps {
    /// Tokens the server will accept in a single input, from llama.cpp's
    /// `default_generation_settings.n_ctx`. Already per-slot — the server reports
    /// the same number it enforces, so `total_slots` must not be divided out.
    pub max_input_tokens: Option<usize>,
    /// Concurrent request slots the server was started with, from llama.cpp's
    /// `total_slots`. Used to size embed concurrency — *not* to adjust
    /// `max_input_tokens`, which is already per-slot (see above).
    pub total_slots: Option<usize>,
    /// Which probe produced `max_input_tokens`. Only set when that probe actually
    /// yielded a usable window — attributing the configured fallback to `/props`
    /// would be a lie.
    pub source: CapsSource,
}

impl ServerCaps {
    /// Fill in whatever this result does not know from `other`, keeping every
    /// field it does.
    ///
    /// The probes answer overlapping but different questions: `/props` is the only
    /// one that ever reports `total_slots`, while `/v1/models` is sometimes the
    /// only one that reports a usable window. Replacing one result wholesale with
    /// the next therefore throws away a slot count that was successfully
    /// discovered — and an unknown slot count silently drops embed concurrency to
    /// serial.
    #[must_use]
    pub fn or(self, other: ServerCaps) -> ServerCaps {
        ServerCaps {
            max_input_tokens: self.max_input_tokens.or(other.max_input_tokens),
            total_slots: self.total_slots.or(other.total_slots),
            // `source` describes where the *window* came from, so it tracks
            // whichever probe actually supplied `max_input_tokens`.
            source: if self.max_input_tokens.is_some() {
                self.source
            } else {
                other.source
            },
        }
    }

    /// Build from a raw token count, treating 0 as unknown. A zero budget is not
    /// usable and must never reach the chunker. `total_slots` is left unset —
    /// only `parse_props` can populate it.
    fn from_tokens(n_ctx: usize, source: CapsSource) -> Self {
        let max_input_tokens = (n_ctx > 0).then_some(n_ctx);
        Self {
            max_input_tokens,
            total_slots: None,
            source: if max_input_tokens.is_some() {
                source
            } else {
                CapsSource::None
            },
        }
    }
}

/// Parse a `GET /props` body. Anything unexpected — missing field, HTML error
/// page, a zero value — yields empty caps rather than an error: capability
/// discovery is best-effort by design. A `llama serve` supervisor lands here with
/// `n_ctx: 0`, which is exactly the "unknown" case.
pub fn parse_props(body: &str) -> ServerCaps {
    #[derive(serde::Deserialize)]
    struct Props {
        #[serde(default)]
        default_generation_settings: Settings,
        #[serde(default)]
        total_slots: usize,
    }
    #[derive(serde::Deserialize, Default)]
    struct Settings {
        #[serde(default)]
        n_ctx: usize,
    }
    let parsed = serde_json::from_str::<Props>(body).ok();
    let n_ctx = parsed
        .as_ref()
        .map_or(0, |p| p.default_generation_settings.n_ctx);
    let total_slots = parsed.as_ref().map_or(0, |p| p.total_slots);
    ServerCaps {
        total_slots: (total_slots > 0).then_some(total_slots),
        ..ServerCaps::from_tokens(n_ctx, CapsSource::Props)
    }
}

/// Parse a `GET /v1/models` body for `model_id`'s context window, the fallback
/// when `/props` is uninformative. Matches on `id` or `aliases[]` only — never on
/// "the single loaded model", because adopting another model's `n_ctx` would be
/// worse than knowing nothing, and the configured id necessarily resolves already
/// (otherwise the embeddings request itself would not route).
pub fn parse_models(body: &str, model_id: &str) -> ServerCaps {
    #[derive(serde::Deserialize)]
    struct Models {
        #[serde(default)]
        data: Vec<Entry>,
    }
    #[derive(serde::Deserialize)]
    struct Entry {
        #[serde(default)]
        id: String,
        #[serde(default)]
        aliases: Vec<String>,
        /// Present only for a loaded model, and only on llama.cpp.
        #[serde(default)]
        meta: Option<Meta>,
    }
    #[derive(serde::Deserialize)]
    struct Meta {
        #[serde(default)]
        n_ctx: usize,
    }
    let Ok(models) = serde_json::from_str::<Models>(body) else {
        return ServerCaps::default();
    };
    let n_ctx = models
        .data
        .iter()
        .find(|e| e.id == model_id || e.aliases.iter().any(|a| a == model_id))
        .and_then(|e| e.meta.as_ref())
        .map_or(0, |m| m.n_ctx);
    ServerCaps::from_tokens(n_ctx, CapsSource::Models)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Trimmed from a real `GET /props` response (llama serve, Qwen3-Embedding-4B).
    /// The sampler `params` block is omitted; `n_ctx` is a sibling of it.
    const REAL_PROPS: &str = r#"{
      "default_generation_settings": { "params": { "top_k": 40 }, "n_ctx": 2048 },
      "total_slots": 4,
      "model_alias": "Qwen/Qwen3-Embedding-4B-GGUF:Q8_0",
      "modalities": { "vision": false }
    }"#;

    /// What `llama serve`'s supervisor answers on its own front port: it describes
    /// itself, not the loaded model, and reports a useless `n_ctx` of 0.
    const ROUTER_PROPS: &str = r#"{
      "role": "router",
      "max_instances": 1,
      "model_alias": "llama-server",
      "model_path": "none",
      "default_generation_settings": { "n_ctx": 0 }
    }"#;

    /// Trimmed from a real `GET /v1/models` through a `llama serve` supervisor.
    /// Only the loaded model carries `meta`; unloaded presets have no such field.
    const REAL_MODELS: &str = r#"{
      "object": "list",
      "data": [
        {
          "id": "Qwen/Qwen3-Embedding-4B-GGUF:Q8_0",
          "aliases": [],
          "status": { "value": "unloaded" }
        },
        {
          "id": "Qwen/Qwen3-Embedding-8B-GGUF:Q8_0",
          "aliases": ["embed-8b"],
          "status": { "value": "loaded" },
          "meta": { "n_vocab": 151665, "n_ctx": 40960, "n_ctx_train": 40960, "n_embd": 4096 }
        }
      ]
    }"#;

    #[test]
    fn reads_per_slot_n_ctx() {
        // n_ctx here is already per-slot: the server reported 2048 with 4 slots
        // and enforced exactly 2048. Do not divide by total_slots.
        assert_eq!(parse_props(REAL_PROPS).max_input_tokens, Some(2048));
    }

    #[test]
    fn missing_field_yields_none() {
        assert_eq!(parse_props(r#"{"total_slots":4}"#).max_input_tokens, None);
    }

    #[test]
    fn non_json_yields_none() {
        // An OpenAI-compatible router may 404 with an HTML page.
        assert_eq!(parse_props("<html>404</html>").max_input_tokens, None);
    }

    #[test]
    fn zero_n_ctx_yields_none() {
        let body = r#"{"default_generation_settings":{"n_ctx":0}}"#;
        assert_eq!(
            parse_props(body).max_input_tokens,
            None,
            "0 is not a usable budget and must not become one"
        );
    }

    #[test]
    fn router_props_yields_none() {
        // The supervisor describes itself, so its n_ctx of 0 must not be adopted.
        assert_eq!(parse_props(ROUTER_PROPS).max_input_tokens, None);
    }

    #[test]
    fn models_reads_loaded_model_context() {
        assert_eq!(
            parse_models(REAL_MODELS, "Qwen/Qwen3-Embedding-8B-GGUF:Q8_0").max_input_tokens,
            Some(40960)
        );
    }

    #[test]
    fn models_matches_on_alias() {
        assert_eq!(
            parse_models(REAL_MODELS, "embed-8b").max_input_tokens,
            Some(40960)
        );
    }

    #[test]
    fn models_ignores_other_entries() {
        // The 4B entry is unloaded and has no meta; asking for it must not borrow
        // the 8B's context window.
        assert_eq!(
            parse_models(REAL_MODELS, "Qwen/Qwen3-Embedding-4B-GGUF:Q8_0").max_input_tokens,
            None,
            "a model without meta must not inherit another model's n_ctx"
        );
    }

    #[test]
    fn models_unknown_id_yields_none() {
        // Never guess from "the only loaded model": a wrong budget is worse than
        // an unknown one.
        assert_eq!(
            parse_models(REAL_MODELS, "some-other-model").max_input_tokens,
            None
        );
    }

    #[test]
    fn models_without_meta_yields_none() {
        // A non-llama.cpp OpenAI-compatible endpoint: standard shape, no meta.
        let body = r#"{"object":"list","data":[{"id":"text-embedding-3-small","object":"model"}]}"#;
        assert_eq!(
            parse_models(body, "text-embedding-3-small").max_input_tokens,
            None,
            "plain `OpenAI` responses must degrade quietly, not error"
        );
    }

    #[test]
    fn models_non_json_yields_none() {
        assert_eq!(parse_models("<html>404</html>", "m").max_input_tokens, None);
    }

    #[test]
    fn models_zero_n_ctx_yields_none() {
        let body = r#"{"data":[{"id":"m","meta":{"n_ctx":0}}]}"#;
        assert_eq!(parse_models(body, "m").max_input_tokens, None);
    }

    #[test]
    fn reads_total_slots() {
        assert_eq!(parse_props(REAL_PROPS).total_slots, Some(4));
    }

    #[test]
    fn missing_total_slots_yields_none() {
        let body = r#"{"default_generation_settings":{"n_ctx":2048}}"#;
        assert_eq!(parse_props(body).total_slots, None);
    }

    #[test]
    fn source_records_which_probe_answered() {
        assert_eq!(parse_props(REAL_PROPS).source, CapsSource::Props);
        assert_eq!(
            parse_models(REAL_MODELS, "embed-8b").source,
            CapsSource::Models
        );
    }

    #[test]
    fn source_is_none_when_the_probe_was_uninformative() {
        // The router answers /props about itself; attributing the fallback budget
        // to /props would misreport where the number came from.
        assert_eq!(parse_props(ROUTER_PROPS).source, CapsSource::None);
        assert_eq!(
            parse_models(REAL_MODELS, "some-other-model").source,
            CapsSource::None
        );
    }

    #[test]
    fn zero_total_slots_yields_none() {
        let body = r#"{"default_generation_settings":{"n_ctx":2048},"total_slots":0}"#;
        assert_eq!(parse_props(body).total_slots, None);
    }

    /// Trimmed from a real `GET /props?model=<id>` through a `llama serve`
    /// supervisor (build b9821, Qwen3-Embedding-8B). Scoping the request with the
    /// model id makes the router proxy to the backend instead of describing
    /// itself: `role` is absent and the real limits appear.
    const SCOPED_PROPS: &str = r#"{
      "default_generation_settings": { "params": { "top_k": 40 }, "n_ctx": 40960 },
      "total_slots": 4,
      "model_alias": "Qwen/Qwen3-Embedding-8B-GGUF:Q8_0",
      "model_path": "/models/qwen3-embedding-8b-q8_0.gguf"
    }"#;

    #[test]
    fn model_scoped_props_reports_the_backends_real_limits() {
        // The router DOES proxy /props — it just needs ?model=. This is the shape
        // that makes both the context window and the slot count discoverable in
        // one probe, where the unscoped call yields neither.
        let caps = parse_props(SCOPED_PROPS);
        assert_eq!(caps.max_input_tokens, Some(40960));
        assert_eq!(caps.total_slots, Some(4));
        assert_eq!(caps.source, CapsSource::Props);
    }

    #[test]
    fn or_fills_gaps_without_discarding_what_is_known() {
        // /props can answer only half the question (slots, but n_ctx 0) while
        // /v1/models answers the other half (n_ctx, never slots). Replacing one
        // wholesale with the other loses the slot count, which silently drops
        // embed concurrency to serial.
        let props = parse_props(r#"{"default_generation_settings":{"n_ctx":0},"total_slots":4}"#);
        let models = parse_models(REAL_MODELS, "embed-8b");

        let merged = props.or(models);

        assert_eq!(merged.total_slots, Some(4), "slot count must survive");
        assert_eq!(merged.max_input_tokens, Some(40960));
        assert_eq!(
            merged.source,
            CapsSource::Models,
            "the window came from /v1/models, so that is what to report"
        );
    }

    #[test]
    fn or_keeps_the_receivers_values_when_it_already_knows_them() {
        let complete = parse_props(SCOPED_PROPS);
        let merged = complete.or(parse_models(REAL_MODELS, "embed-8b"));
        assert_eq!(merged.max_input_tokens, Some(40960));
        assert_eq!(merged.total_slots, Some(4));
        assert_eq!(merged.source, CapsSource::Props, "the first answer wins");
    }
}
