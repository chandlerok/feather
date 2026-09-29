//! The serving path hosted by a plain Rust process.
//!
//! One half of the measurement in `docs/serving-transport.md`: the same Flight service, the same
//! store, the same runtime, hosted by Rust rather than by a Python process. Whatever the Python
//! host costs, it is the difference between this and `serve_host_python.py`.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use feather_core::online::fjall::FjallStore;
use feather_serve::{ResolvedService, ServiceSpec, serve};

#[path = "bench_views.rs"]
mod bench;
use bench::{ENTITY_NAME, PROJECT, SERVICE, bench_views, requested_features};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = PathBuf::from(
        std::env::args()
            .nth(1)
            .unwrap_or_else(|| "/tmp/feather-serve-bench".into()),
    );
    let addr: SocketAddr = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "127.0.0.1:8815".into())
        .parse()?;
    let workers: usize = std::env::args()
        .nth(3)
        .and_then(|a| a.parse().ok())
        .unwrap_or(4);

    let store = Arc::new(FjallStore::open(&dir, 64 << 20, 64 << 20)?);
    let resolved = ResolvedService::resolve(
        PROJECT,
        &ServiceSpec {
            name: SERVICE.into(),
            entity_name: ENTITY_NAME.into(),
            features: requested_features(),
        },
        &bench_views(),
    )
    .expect("the bench service resolves");

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(workers)
        .enable_all()
        .build()?;

    eprintln!("[rust host] serving on {addr} with {workers} workers");
    // `pending()` rather than `async {}`: a shutdown future that is already ready tells tonic to
    // stop at once, which looks exactly like a server that never bound.
    runtime.block_on(serve(addr, store, resolved, std::future::pending()))?;
    Ok(())
}
