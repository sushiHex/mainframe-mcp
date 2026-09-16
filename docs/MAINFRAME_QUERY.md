# Mainframe Query Guide

Mainframe v2 is a local Markdown search service shared by configured MCP
clients. Search it when a repository note or prior captured decision may answer
the current task, then inspect the cited source before relying on it.

## Agent instructions

This is a small, pasteable policy for an agent configuration:

```markdown
## Mainframe
- Search Mainframe when local project knowledge may answer the task.
- Read the returned source before treating a result as evidence.
- When asked to retain context, capture durable decisions, constraints, and follow-up work.
- Search captures only when prior session notes are relevant (`include_sessions: true`).
```

Use terms that identify the implementation, configuration, error, or decision
you need. If the first result set is not useful, revise the terms, inspect a
few more results, or read the cited files. Mainframe searches every project in
its configured scope; a result can therefore come from another configured
project. The search tool has no `project` or `source` filter, so use each
result's `project`, `source_type`, and `file` fields to assess whether it is in
the right context.

## Search

The full MCP endpoint exposes `search`. Its request arguments are:

```json
{
  "query": "index configuration drift rebuild",
  "limit": 3,
  "include_sessions": false,
  "response_format": "concise"
}
```

`query` is required. `limit` defaults to 3 and is constrained to 1–10.
`include_sessions` defaults to `false`: captures are deliberately excluded
from ordinary retrieval. `response_format` is `concise`, `detailed`, or
`full`; concise is the default.

Every response contains `results` and `reranked` — a boolean answering the
caller's real question, whether `rerank_score` is a ranking signal. It is
`false` when nothing ranked the results — either `results` is empty, or the
reranker is administratively disabled (`reranker.enabled: false`), in which
case every `rerank_score` is `0.0` and reflects candidate order only. An
empty result set also carries `guidance`.
Each result always has `file`, `heading`, `rerank_score`, and a short
`snippet`. `rerank_score` orders the retrieved candidates within that
response. It is useful for comparison there, but it is not a claim that a
passage is true, current, applicable, or sufficient to answer the task.
Models load on demand: the first search after a restart may be slow, but a
search always ranks its results or fails — a returned result is always a
ranked result.

For citations or close review, request `detailed`:

```json
{
  "query": "capture lane indexing",
  "limit": 5,
  "response_format": "detailed"
}
```

Detailed results add `text` (up to 500 characters), `truncated`,
`char_start`, `char_end`, `score`, `source_type`, and `project`. They may also
include 1-based `line_start` and `line_end`. Line spans are omitted when the
file changed after indexing or the passage cannot be verified, so do not infer
a line citation from character offsets alone. Read the current `file` before
citing it. Use `full` only when the complete indexed chunk is needed; it
returns untruncated `text` while retaining the detailed fields.

## Capture and recall

On the full endpoint, `capture` writes one immutable, secret-scrubbed Markdown
note. Give it a useful title and dense, durable content rather than a raw
transcript:

```json
{
  "content": "## Summary\nIndexed documents must be Markdown.\n\n## Decision\nUse an explicit project allowlist.",
  "title": "Index scope decision",
  "project": "example-project",
  "kind": "decision"
}
```

`project` is optional (omitting it uses `_adhoc`); `session_id` and `cwd` are
optional; `kind` is `session`, `decision`, or `note` and defaults to `session`.
The daemon watcher normally indexes a new capture after its debounce window.
If it is not visible when needed, call `maintain` with `{"action":"rescan"}`;
the same action may take `{"project":"example-project"}` to rescan that
project. Search it later with `include_sessions: true`.

## Status and recovery

`status` takes no arguments and reports model and index state, project scope,
pending work, and recent events. Use it to diagnose unavailable models,
an incomplete queue, or index drift. The full endpoint also exposes
`maintain`: `index`/`rescan`, `optimize`, and `reload`; the read-only `/mcp/ro`
endpoint offers only `search` and `status`.

An embedding or chunking configuration change can require an offline rebuild.
Changing only the reranker does not: it reorders the existing first-stage
candidates. For an embedding or chunking change, stop the daemon, then run
`mainframe rebuild`, then start it again with `mainframe serve`.
See the [v2 daemon guide](V2.md#index-ownership-and-recovery) for the exact
recovery sequence. This replaces the legacy v1 tool list; see the
[migration guide](PREVIEW_RELEASE.md) when working with an older setup.
