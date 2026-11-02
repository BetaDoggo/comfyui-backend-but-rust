use std::sync::Arc;

use clap::Parser;
use comfy_server::config::Config;
use comfy_server::mock_engine::MockEngine;
use swarm_engine_api::Engine;

fn make_engine(cfg: &Config) -> anyhow::Result<Arc<dyn Engine>> {
    match cfg.engine.as_str() {
        "mock" => Ok(Arc::new(if cfg.mock_native_only { MockEngine::native_only(cfg.mock_step_ms) } else { MockEngine::new(cfg.mock_step_ms) })),
        #[cfg(feature = "native")]
        "native" => {
            // Serves Krea 2, Anima and MOSS Image; the checkpoint picks which.
            let options = native_engine::NativeOptions { reserve_vram_mb: cfg.reserve_vram, max_side: cfg.max_side, ..Default::default() };
            Ok(Arc::new(native_engine::NativeEngine::new(&options)?))
        }
        #[cfg(not(feature = "native"))]
        "native" => anyhow::bail!("this build has no native engine; rebuild with --features native"),
        other => anyhow::bail!("unknown --engine {other} (mock | native)"),
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
