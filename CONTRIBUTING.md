# Contributing to mainframe-mcp

Contributions are welcome — from humans and from coding agents. This document
is written for both; if you are an autonomous agent, treat it as binding
instructions for any PR you open here.

## Public development and private data

**This public repository is the primary development home.** Code, issues,
reviews, and releases belong here. Branch from `main` and merge reviewed pull
requests directly into public `main`; preserve its history.

The private repository stores non-public data, research, captures, and
corpus-specific experiment records. It is not the upstream for public code.
When bringing over a code fix, review and transfer only the required public
files onto a public branch. Never merge private Git history or force-push a
replacement snapshot over public `main`. The old mirror publisher is retired.

**Every public push requires an audit first**, including branch pushes before
opening a PR. Follow [the public audit procedure](docs/PUBLIC_AUDIT.md), review
all findings, and record the exact audited commit. CI repeats the automated
checks; it cannot replace the review before publication.

## Dev setup

```bash
pip install -e .[test]
python -m pytest tests/
```

**No GPU is needed to contribute.** The test suite is deliberately GPU-free
(deterministic fake embedders + a real LanceDB on disk) and must stay that
way: new tests must not load models, hit the network, or assume CUDA. CI runs
exactly `python -m pytest tests/` on a CPU-only Linux runner — if it doesn't
pass there, it doesn't merge.

The `package-install` gate also requires clean wheel installations on Windows
and Linux with Python 3.10 and 3.14. `scripts/check_install.py` imports the runtime
and checks configuration and entry points outside the checkout, without pytest's
source-path injection. Keep that check GPU-free and preserve its stable gate name
when adjusting the matrix.

Use temporary directories for indexes and captures. The v2 suite also drives
real filesystem watchers; its `slow` tests run by default. During iteration,
`python -m pytest tests/v2/ -m "not slow"` skips those timing checks. See
[the v2 guide](docs/V2.md) for service setup; a live deployment is not required
to contribute.

## What a PR must include

1. **Tests, written test-first.** Bug fixes need a regression test that fails
   without the fix (say so in the PR description — "RED before GREEN" is the
   norm here). Features need tests exercising the real code paths through the
   GPU-free harness (`tests/conftest.py` shows the pattern).
2. **A green suite locally**: `python -m pytest tests/`.
3. **Honest claims.** Never assert a retrieval-quality improvement without
   measurement (see below). Never claim "tests pass" without running them.
4. **No secrets, no personal data.** No absolute personal paths, private
   corpora, or real API keys even as test fixtures. Use obviously synthetic
   examples. Include the public-audit result; tests and documentation are
   subject to the same review as source code.
5. **Docs updated** when behavior changes: `CLAUDE.md` is the agent-facing
   project brief — keep it truthful; `README.md` for user-facing changes.

## Retrieval-quality changes are special

Changes that can move search quality (chunking, embedding prefixes, reranker
scoring, candidate pool, tier boosts, dedup) **cannot be validated by CI** —
they need the eval harness against a real corpus, which only the maintainer's
deployment has.

The house rule, learned the hard way: **leaderboards and model cards have
inverted against local measurement four separate times in this project.**
So:

- Say clearly in the PR that the change is retrieval-affecting.
- Don't claim a quality win you haven't measured. Describe the *mechanism* and
  the *expected* effect; the maintainer runs the regression gate
  (`eval/evaluate.py --live --min-score <baseline>`) before merging.
- If you ran the harness on your own corpus, log methodology and numbers
  following `eval/results.template.md` in the PR description. Numbers are
  corpus-specific; treat them as evidence, not proof.

Behavior-preserving changes to *measured* code paths deserve extra care: some
hot paths are deliberately bit-stable (e.g. the INT8 reranker batch order —
see the comment in `reranker.py`); comments saying "do not change, measured"
mean exactly that.

## Conventions

- Python 3.10+; match the surrounding code's style, naming, and comment
  density. Comments state non-obvious constraints, not narration.
- Commit titles follow "Component: what changed" (e.g. "Capture extractor: URLs
  are not file paths") — no Conventional-Commits prefixes.
- One logical change per PR; reference the issue if one exists.
- Windows and POSIX both matter (the maintainer develops on Windows, CI runs
  Linux): use `pathlib`, never hardcode path separators or drive letters.

## Where things live

| Path | What it is |
|------|-----------|
| `src/mainframe_mcp/` | The server — one module per concern (see `CLAUDE.md` for the map) |
| `src/mainframe/` | v2 preview: indexing/search core, memory lanes, daemon service, CLI/MCP adapters |
| `tests/` | GPU-free suite; `conftest.py` has the fake-embedder harness |
| `tests/v2/` | v2 regressions, real temporary stores, watcher and transport coverage |
| `eval/` | The measurement harness + experiment-log template |
| `configs/` | VRAM-tier presets |

## Reporting issues

Issues are welcome on the public repo. For suspected retrieval-quality
regressions, include your corpus size, config deltas from defaults, and
example queries with expected-vs-actual results — "search feels worse" isn't
actionable; a reproducible query is.
