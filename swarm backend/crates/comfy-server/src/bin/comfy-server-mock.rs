//! `comfy-server` without the `native` feature: only `--engine mock`. It is its own binary so that `cargo test` and
//! plain builds, which leave the feature out, never overwrite the native `comfy-server.exe` that SwarmUI starts.

include!("../main.rs");
