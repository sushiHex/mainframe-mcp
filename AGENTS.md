# Repository Guidelines

## Development Home

The public `sushiHex/mainframe-mcp` repository is the primary home for code, issues, reviews, and releases. Branch from public `main` and merge reviewed pull requests there. The private repository stores non-public material; never merge its history into this repository or replace public history with a snapshot.

## Project Structure

`src/mainframe_mcp/` contains the MCP server, retrieval pipeline, storage, and memory features. `tests/` contains GPU-free pytest coverage and shared fixtures. `configs/` provides hardware presets, `docs/` holds public documentation, and `eval/` contains evaluation tools and synthetic examples. Private corpora and experiment records belong outside this repository.

## Development Commands

- `python -m pip install -e ".[test]"`: install editable packages and test dependencies.
- `python -m pytest tests/ -q`: run the complete GPU-free suite.
- `python -m pytest tests/test_secrets.py -q`: run focused secret-scrubbing tests.
- `mainframe serve`: start the current v2 daemon (see `docs/V2.md`); `python -u -m mainframe_mcp.server` starts the legacy v1 stdio server only (see `docs/PREVIEW_RELEASE.md`).
- `python -m pip wheel . --no-deps -w dist`: build a distribution wheel.

## Style and Testing

Use four-space indentation, `snake_case` functions and modules, and `PascalCase` classes. Match nearby annotations and formatting; no formatter is configured. Use `pathlib` for filesystem operations. Write regression tests before fixes, using deterministic fake models and temporary LanceDB stores. Tests must not load real models, access the network, or require CUDA. Retrieval-quality claims require measured evidence.

## Public Push Audit

Before every public push, follow [docs/PUBLIC_AUDIT.md](docs/PUBLIC_AUDIT.md). Audit the exact outgoing commits, their messages, complete file contents, intermediate history, and artifacts. Include tests and documentation. Check credentials, personal paths, private project details, captures, corpora, and generated data. Review each scanner finding; never exempt entire test directories. Record the audited commit and repeat the audit after changes. Do not push unresolved findings or private Git ancestry.

## Commits and Pull Requests

Use concise `Component: what changed` titles. Keep changes focused, explain the problem and mechanism, link relevant issues, and report actual validation. Complete the PR template, including public-audit evidence. See [CONTRIBUTING.md](CONTRIBUTING.md) for the full workflow.
