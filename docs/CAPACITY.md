# Concurrent traffic: measured operating range

## 2026-10-13: 1, 4 and 8 concurrent clients on one daemon. Supported to 4, at the edge at 8

**Hypothesis.** One daemon serves several agents at once, and searches serialize on its single model lock. Latency should therefore grow about linearly with the number of clients, while throughput stays flat. The targets below were proposed on #21 *before* the run.

**Method.**

```
mainframe validate --output <new dir> --clients 1,4,8 --load-seconds 60
```

- The validation daemon is fresh, on a synthetic corpus and scratch state, alongside the production daemon.
- Each phase runs independent HTTP clients over four fixed queries: a 60 s closed loop, then a three-round burst.
- A file write and a capture run during each closed loop, to measure index and capture visibility while the GPU is busy.
- `reconnect` repeats the widest phase against the restarted process.

**Setup.**
- RTX 3090 24 GB, driver 596.49, Windows 11, Python 3.14.
- `mainframe` 2.0.0a2, Harrier OSS v1 0.6B + Qwen3-Reranker-0.6B, both BF16.
- A vram-mcp reservation was held, with no training or other model load running.

| phase | clients | completed / requests | req/s | p50 | p95 | max | p50 ÷ 1-client p50 | errors · timeouts · overload |
|---|---|---|---|---|---|---|---|---|
| warm | 1 | 301 / 301 | 5.00 | 0.19 s | 0.31 s | 0.46 s | 1.00 | 0 · 0 · 0 |
| warm | 4 | 340 / 340 | 5.61 | 0.70 s | 0.90 s | 1.25 s | 3.74 | 0 · 0 · 0 |
| warm | 8 | 312 / 312 | 5.11 | 1.53 s | 1.93 s | 2.06 s | **8.22** | 0 · 0 · 0 |
| reconnect | 8 | 336 / 336 | 5.53 | 1.40 s | 1.77 s | 2.00 s | 7.50 | 0 · 0 · 0 |

The figures above are for the closed loop. Burst p95 was 0.19, 0.74, 1.62 and 1.33 s respectively.

**Other measurements, across all phases**
- Health checks: 0 failures in 177 probes, p95 0.005 s.
- Visibility under load: 5.0–7.1 s for a written file, 4.9–5.7 s for a capture.
- Memory: daemon RSS at most 2.05 GB; host available memory never below 10 GB; driver-reported free VRAM never below 21.6 GB.
- Startup: ready in 3.7 s, first search 3.9 s (it loads the reranker), restart ready in 4.6 s.
- Uncited answers (11–23 per phase) are the probe file being deleted mid-search. A citation is omitted for a file that moved, never fabricated.

**Targets and verdict.** All targets were met except one:

| target | verdict |
|---|---|
| 1-client p50 ≤ 1.5 s and p95 ≤ 3 s | met |
| p50 at N clients ≤ N × the 1-client p50 | met at 4 (3.74), and in the 8-client reconnect phase (7.50); **exceeded at 8 in the warm phase (8.22, 3% over)** |
| p95 ≤ 1.5 × the scaled p50 | met at every count |
| 0 errors, timeouts and overloads; 0 health failures, p95 ≤ 0.5 s | met |
| daemon RSS ≤ 6 GB, ≥ 4 GB host memory available | met |
| visibility ≤ debounce + 30 s | met |

**Decision.**
- **Supported: up to 4 concurrent agents**, at under a second for a median search.
- **8 agents work** (no errors, p95 under 2 s), but sit at the edge of linear scaling: each one waits for the other seven, so treat 8 as the practical ceiling rather than a comfortable operating point.
- Throughput is about 5–5.6 searches/s regardless of client count. More clients means more waiting, not more work done.

**Recommended settings.**
- Keep the defaults.
- Set `service.prewarm: true` when agents start their sessions together, so the first search does not pay the model load.

**Not measured.**
- *Queue wait:* searches wait on an in-process lock that nothing counts. It is inferred above from p50 scaling.
- *PyTorch allocated/reserved memory:* only driver VRAM is exposed.
- *Desktop responsiveness:* not assessed for this run.

The corpus is synthetic, so this measures concurrency, not retrieval quality. Raw reports stay local.
