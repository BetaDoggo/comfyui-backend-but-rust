use std::sync::Arc;

use clap::Parser;
use comfy_server::config::Config;
use comfy_server::mock_engine::MockEngine;
use swarm_engine_api::Engine;

fn make_engine(cfg: &Config) -> anyhow::Result<Arc<dyn Engine>> {
    match cfg.engine.as_str() {
        "mock" => Ok(Arc::new(if cfg.mock_native_only { MockEngine::native_only(cfg.mock_step_ms) } else { MockEngine::new(cfg.mock_step_ms) })),
        #[cfg(feature = "krea2")]
        "krea2" => {
            // Serves Krea 2 and Anima; the checkpoint picks which.
            let options = krea2_engine::Krea2Options { reserve_vram_mb: cfg.krea2_reserve_vram, max_side: cfg.krea2_max_side, ..Default::default() };
            Ok(Arc::new(krea2_engine::Krea2Engine::new(&options)?))
        }
        #[cfg(not(feature = "krea2"))]
        "krea2" => anyhow::bail!("this build has no krea2 engine; rebuild with --features krea2"),
        other => anyhow::bail!("unknown --engine {other} (mock | krea2)"),
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info,comfy_server=info".into()))
        .init();
    let cfg = Config::parse();

    let engine = make_engine(&cfg)?;

    let (addr, handle) = comfy_server::spawn(&cfg, engine).await?;
    tracing::info!("comfy-server listening on http://{addr} ({} engine)", cfg.engine);
    tracing::info!("model roots: {:?}", cfg.models_roots);
    tracing::info!("in SwarmUI: Server -> Backends -> add 'ComfyUI API By URL' with address http://{addr}");
    handle.await?;
    Ok(())
}
