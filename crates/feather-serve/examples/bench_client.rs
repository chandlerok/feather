//! Load generator for the serving-transport measurement.
//!
//! A separate process on purpose. A client inside the server's process would contend for the
//! same cores, and a client in Python would hold the GIL it is trying to measure the absence of.
//! So this is Rust, it is separate, and its cost is visible in the numbers as a ceiling rather
//! than as a floor.
//!
//! Reports requests per second and latency percentiles, because the throughput figure alone
//! cannot tell a server that is fast from one that is queuing.

use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use arrow_flight::Ticket;
use arrow_flight::flight_service_client::FlightServiceClient;
use tonic::transport::Channel;

use feather_serve::ReadRequest;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let addr: SocketAddr = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "127.0.0.1:8815".into())
        .parse()?;
    let concurrency: usize = std::env::args()
        .nth(2)
        .and_then(|a| a.parse().ok())
        .unwrap_or(8);
    let seconds: u64 = std::env::args()
        .nth(3)
        .and_then(|a| a.parse().ok())
        .unwrap_or(10);
    let entities_per_request: usize = std::env::args()
        .nth(4)
        .and_then(|a| a.parse().ok())
        .unwrap_or(1);

    let client = FlightServiceClient::new(
        Channel::from_shared(format!("http://{addr}"))?
            .connect()
            .await?,
    );

    // Build the tickets up front so the client's own JSON encoding is not inside the timed loop.
    let mut tickets: Vec<Ticket> = Vec::with_capacity(entities_per_request);
    for _ in 0..entities_per_request {
        let request = ReadRequest {
            service: "serving".into(),
            entities: (0..8)
                .map(|e| {
                    feather_core::key::encode_entity_key(&[b"u1", format!("{:012}", e).as_bytes()])
                        .expect("key")
                })
                .collect(),
        };
        tickets.push(Ticket {
            ticket: serde_json::to_vec(&request)?.into(),
        });
    }

    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let completed = Arc::new(AtomicU64::new(0));
    let mut handles = Vec::with_capacity(concurrency);

    for worker in 0..concurrency {
        let mut client = client.clone();
        let tickets = tickets.clone();
        let stop = Arc::clone(&stop);
        let completed = Arc::clone(&completed);
        let _ = worker;
        handles.push(tokio::spawn(async move {
            let mut latencies: Vec<u64> = Vec::with_capacity(4096);
            while !stop.load(Ordering::Relaxed) {
                for ticket in &tickets {
                    if stop.load(Ordering::Relaxed) {
                        break;
                    }
                    // Only `do_get` is timed. `get_flight_info` is the metadata handshake a
                    // client does once per descriptor, not a serving call, and putting it in
                    // the loop would measure a round trip the real path does not make per read.
                    let started = Instant::now();
                    let mut stream = client
                        .do_get(Ticket {
                            ticket: ticket.ticket.clone(),
                        })
                        .await?
                        .into_inner();
                    while stream.message().await?.is_some() {}
                    latencies.push(started.elapsed().as_nanos() as u64);
                    completed.fetch_add(1, Ordering::Relaxed);
                }
            }
            Ok::<_, Box<dyn std::error::Error + Send + Sync>>(latencies)
        }));
    }

    tokio::time::sleep(Duration::from_secs(seconds)).await;
    stop.store(true, Ordering::Relaxed);

    let mut all: Vec<u64> = Vec::new();
    for handle in handles {
        match handle.await {
            Ok(Ok(mut l)) => all.append(&mut l),
            Ok(Err(e)) => eprintln!("worker error: {e}"),
            Err(e) => eprintln!("worker panicked: {e}"),
        }
    }
    all.sort_unstable();

    let done = completed.load(Ordering::Relaxed);
    let rate = done as f64 / seconds as f64;
    let per_entity = rate * entities_per_request as f64;
    println!(
        "host                {}",
        std::env::var("HOST_LABEL").unwrap_or_else(|_| "unset".into())
    );
    println!(
        "concurrency         {concurrency} clients, {entities_per_request} entities per request"
    );
    println!("requests            {done} in {seconds}s = {rate:.0} req/s");
    println!("entity reads        {per_entity:.0} reads/s");
    println!(
        "latency p50         {:.1}us",
        pct(&all, 0.50) as f64 / 1000.0
    );
    println!(
        "latency p99         {:.1}us",
        pct(&all, 0.99) as f64 / 1000.0
    );
    Ok(())
}

fn pct(sorted: &[u64], p: f64) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    sorted[(((sorted.len() - 1) as f64) * p).round() as usize]
}
