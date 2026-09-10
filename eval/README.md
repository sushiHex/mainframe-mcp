# Retrieval Evaluation

Evaluate retrieval against your own corpus and expected answers. The GPU-free
pytest suite checks behavior; this harness measures retrieval quality.

## Select the implementation

```bash
# v2: running daemon, or configured index when it is stopped
python -u eval/evaluate.py --live

# v2: require the daemon and avoid loading models here
python -u eval/evaluate.py --daemon

# v1: original stdio server's index
python -u eval/evaluate_v1.py --live
```

Use the same configuration and query set for comparisons. Set `MAINFRAME_CONFIG`
for the v2 deployment being evaluated. The legacy v1 harness expects the default
`~/.claude/mainframe` state directory. Never run a second GPU-heavy evaluation
alongside an existing model allocation.

## v2 modes

| Option | Behavior |
|---|---|
| Default / `--live` | Uses the running daemon; otherwise opens the configured v2 index, or builds a temporary index if absent |
| `--daemon` | Requires the daemon; scores HTTP search with full chunk text |
| `--db PATH` | Opens an existing populated v2 index; refuses v1 schemas |
| `--rebuild` | Replaces `eval/.eval-lancedb` with a deterministic rebuild |
| `--corpus-manifest PATH` | Restricts a rebuild to existing non-session files in a v1 manifest |
| `--dump-candidates PATH` | Saves raw candidate identities for deterministic comparisons |
| `--min-score NUMBER` | Exits 1 below the supplied composite-score baseline |

`--db` and `--rebuild` refuse while the configured daemon is running. Explicit
`--in-process` and candidate dumps load their own models; stop other GPU-heavy
work first. Compare discovery with a v1 manifest using
`python -u eval/corpus_identity.py path/to/.manifest.json`. The gate fails when
a still-existing manifest file is missing from the v2 scan; new files and
content changes are reported separately.

## Queries, results, and measurement

Copy `eval/test_queries.example.json` to gitignored `eval/test_queries.json`
and supply queries, expected files, and optional expected text. The example
set demonstrates the format; its scores are not a corpus baseline.

Runs save local JSON under `eval/results/`. Review with
`python eval/history.py --best` or `python eval/history.py --last 5`.

The composite is **40% MRR + 30% hit@1 + 30% text_match@1**. Both v2 paths
use the same top-three rank window. Daemon `hit_at_5` is `null` because ranks
four and five are not requested. Text matching uses complete chunks.

Change one variable at a time and record corpus identity, configuration,
models, timing, and repeated runs against the same baseline. Follow
`results.template.md`; keep corpus-specific logs in gitignored `results.md`.
Query sets, candidate dumps, and results can expose private paths and project
details. Audit any proposed publication before uploading it.
