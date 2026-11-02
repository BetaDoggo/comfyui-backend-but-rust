//! A ComfyUI-API-compatible server that SwarmUI can add as a "ComfyUI API By URL"
//! backend. It validates and executes ComfyUI prompts by lowering each node into
//! calls on a [`swarm_engine_api::Engine`].

pub mod config;
pub mod encode;
pub mod env;
pub mod executor;
pub mod graph;
pub mod mock_engine;
pub mod models;
pub mod nodes;
pub mod queue;
pub mod routes;
pub mod schema;
pub mod todo;
pub mod value;
pub mod ws;

use std::net::SocketAddr;
use std::sync::Arc;

use swarm_engine_api::Engine;

use crate::config::Config;
use crate::env::ServerEnv;
use crate::executor::Executor;
use crate::nodes::NodeRegistry;
use crate::queue::PromptQueue;
use crate::routes::AppState;
use crate::ws::Hub;

/// Builds the app state and starts the worker thread. Returns the axum router.
pub fn build(cfg: &Config, engine: Arc<dyn Engine>) -> anyhow::Result<axum::Router> {
    let mut env = ServerEnv::from_config(cfg)?;
    env.attention_backends = engine.info().attention_backends;
    let env = Arc::new(env);
    let registry = Arc::new(NodeRegistry::standard(&env, engine.as_ref()));
    let queue = Arc::new(PromptQueue::new(cfg.max_history));
    let hub = Arc::new(Hub::default());
    let executor = Executor::new(registry.clone(), engine.clone(), env.clone(), queue.clone(), hub.clone());
    std::thread::Builder::new().name("engine-worker".into()).spawn(move || executor.run_forever())?;
    Ok(routes::router(AppState { registry, env, queue, hub, engine }))
}

/// Binds and serves in the background; returns the bound address (use port 0 in tests).
pub async fn spawn(cfg: &Config, engine: Arc<dyn Engine>) -> anyhow::Result<(SocketAddr, tokio::task::JoinHandle<()>)> {
    let app = build(cfg, engine)?;
    let listener = tokio::net::TcpListener::bind((cfg.listen.as_str(), cfg.port)).await?;
    let addr = listener.local_addr()?;
    let handle = tokio::spawn(async move {
        if let Err(e) = axum::serve(listener, app).await {
            tracing::error!("server error: {e}");
        }
    });
    Ok((addr, handle))
}
