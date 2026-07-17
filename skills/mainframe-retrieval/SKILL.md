---
name: mainframe-retrieval
description: When and how to use the Mainframe knowledge base — recall from your own past research/docs/session notes via semantic search, and (optionally) capture new durable knowledge. Load when a task needs prior decisions, gotchas, or project conventions you may have recorded before.
---

# Using the Mainframe

The Mainframe (`mainframe-mcp`) is a local semantic search index over your
markdown knowledge — `research/`, `docs/`, and root context files (`CLAUDE.md`,
`AGENTS.md`, …) across your repos, plus any session notes you've captured. It is
a **recall aid, not an authority**: it returns what you wrote before, which may
be stale or wrong. It is NOT a task tracker, calendar, or source of truth for
deadlines — treat hits as evidence to verify, not facts to trust blindly.

This skill is optional and host-agnostic: it keeps the query technique out of
the permanent tool-schema prompt and loads it only when retrieval is relevant.

## When to search

- Before answering from training data on anything project-specific — you may
  have researched it already.
- Before a design/tech/gotcha decision: check for a prior conclusion.
- When you hit an error or constraint that feels familiar.

Don't search for general knowledge the model already has, or for things that
change faster than you index (live status, current task state).

## How to phrase a query — this is the whole skill

**Specific technical terms rerank ~0.99; natural-language questions rerank
~0.05.** Use proper nouns, function/class names, exact config keys, error
strings — not sentences.

- GOOD: `LanceDB to_pandas columns kwarg projection workaround`
- GOOD: `Qwen3 reranker yes-no logit native scoring instruction`
- BAD: `how does the reranker work` (no rerank signal)

One topic per query. Two focused searches beat one broad one.

## Reading the result

`search` returns `{"results": [...], "confidence": "high"|"low"|"none",
"guidance": "..."}`.

- **Trust `rerank_score`** (higher = better; >0.85 strong, <0.3 likely noise),
  NOT `score` (raw vector distance).
- `confidence: "none"` → no hit; `"low"` → best score < 0.3. On both, `guidance`
  tells you how to rephrase — follow it (add specific terms) and retry once.
- To read the full source: each hit has `file` + `line_start`/`line_end` — call
  your `read_file(path=file, offset=line_start, ...)` to pull the exact section
  and cite it. (`char_start`/`char_end` are there too for char-oriented tools.)

## Weak-query retry loop

1. Search with specific terms.
2. If `confidence` is `none`/`low`, rephrase per `guidance` (more/different
   exact terms, or `include_sessions=true` to also search raw session captures)
   and retry once.
3. Still nothing → the knowledge isn't indexed. Answer from first principles and
   consider capturing what you learn (below).

## When to capture memory

Capture a durable note (via `capture_memory`, or your host's capture flow) when
you reach a **hard-won, reusable conclusion**: a gotcha, a measured result, a
design decision with rationale. Do NOT capture: routine task chatter,
speculation, secrets, or anything that changes daily. Files are the source of
truth; the index is derived — write the note, let the server index it.
