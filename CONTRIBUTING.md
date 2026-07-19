# Contributing to mainframe-mcp

Contributions are welcome — from humans and from coding agents. This document
is written for both; if you are an autonomous agent, treat it as binding
instructions for any PR you open here.

## The two-repo architecture (read this first)

Development happens in a **private repository** with full history. The public
repo you are looking at is a **squashed snapshot mirror**: its `main` branch is
force-pushed on every release, and its single-commit history is rebuilt each
time.

Consequences for pull requests:

- **PRs against public `main` are never merged directly.** Merging into a
  branch that gets force-replaced would destroy the merge on the next refresh.
- Instead, the maintainer applies your change to the private repo (preserving
  attribution in the commit message), the mirror is refreshed, and your PR is
  closed with a comment when your change ships. Your diff lands; your commit
  SHA doesn't survive.
- Keep PRs small and self-contained — they are applied as patches, so a
  focused diff with tests is far easier to land than a sprawling one.

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

Note for agents reading `CLAUDE.md`: the sections about the live index
(`~/.claude/mainframe`), SessionStart/SessionEnd hooks, VRAM budgets, and
"never run two GPU scripts" describe the **maintainer's deployment**, not your
fork. You cannot and need not reproduce that environment to contribute.

## What a PR must include

1. **Tests, written test-first.** Bug fixes need a regression test that fails
   without the fix (say so in the PR description — "RED before GREEN" is the
   norm here). Features need tests exercising the real code paths through the
   GPU-free harness (`tests/conftest.py` shows the pattern).
2. **A green suite locally**: `python -m pytest tests/`.
3. **Honest claims.** Never assert a retrieval-quality improvement without
   measurement (see below). Never claim "tests pass" without running them.
4. **No secrets, no personal data.** Nothing matching `secrets.py`'s denylist
   patterns, no absolute personal paths, no real API keys even as test
   fixtures (use obviously-fake values like the existing tests do). A CI
   leak-check runs on every PR.
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
| `tests/` | GPU-free suite; `conftest.py` has the fake-embedder harness |
| `eval/` | The measurement harness + experiment-log template |
| `configs/` | VRAM-tier presets |

## Reporting issues

Issues are welcome on the public repo. For suspected retrieval-quality
regressions, include your corpus size, config deltas from defaults, and
example queries with expected-vs-actual results — "search feels worse" isn't
actionable; a reproducible query is.
