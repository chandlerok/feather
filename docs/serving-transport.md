# The serving transport: why it is in Rust, measured

Status: **measurement, not a decision record.** One question was asked and answered: if a user's
own process hosts the serving surface, does the Python in that process cost anything?

## The question

`FeatureStore.serve()` makes the caller the server. Python imports the module and blocks. If the
request path were Python, the GIL would serialise every request, the answer would be process
workers, and more processes is exactly what an embedded store cannot be opened by: a fjall
database takes an exclusive lock, so one directory has one owner.

So the transport is in `feather-serve` and the request path is entirely Rust. That is a design
decision, and the measurement below is what it is worth.

## What was measured

The same Flight service, over the same store, on the same 4-vCPU host with the same tokio worker
count. The only difference is what hosts the process: a plain cargo binary, or the same server
reached through pyo3 from a Python interpreter. The store held 200,000 entities x 4 views of 8
float64 features, 142MB on disk, written through the real `encode_batch` and carrying a real
freshness timestamp. A third process, also Rust, generated load; it is Rust and separate on
purpose, because a client inside the server would contend for the same cores and a Python client
would hold the GIL the measurement is about the absence of.

## Result

8 clients, 8 entities per request, 4 tokio workers, 12 seconds:

| host                        | requests/s | entity reads/s | p50     | p99      |
| --------------------------- | ---------- | -------------- | ------- | -------- |
| Rust (plain cargo binary)   | 8,014      | 64,111         | 905.7us | 2486.5us |
| Python (pyo3, GIL released) | 7,270      | 58,159         | 986.0us | 2963.9us |

Python hosts at **0.91 of the Rust host**. Across worker and concurrency sweeps the ratio sat
between 0.82 and 0.92, so the honest range is **8 to 18% of throughput**, which is contention for
the same cores rather than a GIL. The failure mode the design guards against, a few thousand
requests per second, does not appear.

An earlier run of this measurement, before the read path went through `read_entities`, reported
11,491 and 10,273 requests per second. The lower figures above are the correct ones: that run
returned the store's raw bytes, and this one decodes each view's vector and applies the read-time
TTL check, which is what serving actually does.

## The GIL, directly

Throughput alone would not distinguish "the GIL is released" from "the GIL is held but the server
does not need it". So the Python host ran a second thread that counted in a loop and reported
while serving:

```text
[python host] serving on 127.0.0.1:8841 with 4 workers, GIL released across serve()
[python host] python thread progressed: 500 ticks
...
[python host] python thread progressed: 11000 ticks
```

11,000 ticks across 12 seconds, while the same process handled 87,239 requests. The GIL is
released across the blocking `serve`, and a user's other threads keep running.

## What these numbers are not

- **The absolute figures are depressed by client contention.** The generator shares four cores
  with the server, which is why p50 is around 900us rather than the single-digit microseconds a
  warm store read costs. The rust-to-python ratio is the finding; the absolute rates are not a
  capacity claim.
- **The value codec is not in this path.** The stored rows are written through the real
  `encode_batch`, so the read decodes them for real, but the response is checked for shape rather
  than compared against a source of truth.
- **One host, one box.** Nothing here says anything about what happens with a replica per pod,
  which is where the consistency questions live.

## What follows

`serve()` can be a method on a user's `FeatureStore`, the serving process can be the
application, and the store can be embedded. The three hold together: the feature server is the
only thing that opens the directory, so there is one writer set, and a push is visible to every
reader without a fan-out.

## Reproducing

Three of the four pieces ship, and the fourth does not. The runnable part:

```bash
cargo build --release -p feather-serve --examples
# fill_store <entities> <views> <dir>
./target/release/examples/fill_store 200000 4 /tmp/feather-serve-bench
# serve_rust <dir> <addr> <tokio workers>
./target/release/examples/serve_rust /tmp/feather-serve-bench 127.0.0.1:8815 4 &
# bench_client <addr> <clients> <seconds> <entities per request>
./target/release/examples/bench_client 127.0.0.1:8815 8 12 8
```

Every argument is positional. There is no runner script and there are no environment variables;
an earlier draft of this section showed both, and neither exists.

**The Python host does not ship.** The rows above labelled "Python (pyo3, GIL released)" and
the `python thread progressed` log came from a throwaway pyo3 module built against this crate by
path, living outside the repository. It is not in the diff, so **the Python-host column cannot be
reproduced from what is here.** The part that is reproducible is that `serve_rust` is the same
`feather_serve::serve` call a Python `FeatureStore.serve()` makes, differing only in the host
process, which is why comparing the two isolates the cost of hosting in Python.

Reproducing it needs a pyo3 module that calls `feather_serve::serve`, which is what
`crates/feather-py` now does through `_core.serve`. A measurement harness built the same way
would belong here rather than in `/tmp`; until one lands, treat the Python column as the number
it is: a reading from a prototype that this change does not ship.

The point-read figures in the comparison above come from a separate standalone bench that is also
not in this diff. The 5.5us warm and ~150us cold readings for the store are reproducible with
`fill_store` plus `bench_client`; the Valkey figures are Feast's published numbers for a Redis
Cluster, not a measurement taken here.
