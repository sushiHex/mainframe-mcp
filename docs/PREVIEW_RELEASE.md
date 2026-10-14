# 2.0.0b1

**2.0.0b1** is the v2 daemon with the v1 server removed: Harrier 0.6B native
(1024-dim) embedding and Qwen3-Reranker-0.6B, unchanged from 2.0.0a2, so an
a2 index is reused as is — no rebuild.

## Changes since 2.0.0a2

- **Readiness and startup.** `/healthz` and `status` report `readiness`
  (`ready`, `warming`, `idle`, `blocked` with its remedy); `service.prewarm`
  loads the models at start and `up`/`restart` wait for `ready` (#53).
  `down` returns only when the daemon's processes are gone (#59).
- **A daemon that dies comes back.** `mainframe serve` supervises `serve --once`
  and restarts it with backoff; `mainframe down` is never undone (#61).
- **Windows logon task without an elevated shell, and without a console
  window**; a windowless daemon writes what a crash says to `daemon.out` (#56).
- **Keyword index stays consistent.** It is rebuilt before lance's index merge
  could leave it pointing at a dropped fragment (#50).
- **Session capture hook.** `mainframe hook session-end` applies the
  `capture.*` gate and writes one v2 capture (#63).
- **Measured capacity.** `mainframe validate --clients 1,4,8` and the
  [operating range](CAPACITY.md): up to four concurrent agents (#58, #62).
- **Clients.** Codex setup and connection choice documented; two clients
  verified on one daemon (#54).

**Upgrading from 2.0.0a2:** Python 3.11 is now the minimum (3.10 reached
end-of-life, #55). The [v1 removal](#v1-removal) (#60) means five v1-only
configuration keys are now refused at startup; delete them if present.
Re-run `mainframe install-task` once to get the windowless task (a task
registered from an elevated shell can only be replaced from one).

**2.0.0a2** switched the default stack from Qwen3-Embedding-8B to Harrier.
**Breaking change there:** an index built under 2.0.0a1 must be rebuilt offline,
because the default embedder changed (Qwen3-Embedding-8B, 4096-dim, to Harrier,
1024-dim) and its stored vectors no longer share a space with a2's queries. The
index marker records the embedder that built it, so a2 opening an a1 index
refuses both searches and writes (`IndexStaleError`, naming the changed setting
and the remedy) until the rebuild: run `mainframe down`, then `mainframe rebuild`.
Nothing is converted in place.

**2.0.0a1 (historical)** shipped the original v1 server and opt-in v2 daemon
with Qwen3-Embedding-8B (4096-dim) + Qwen3-Reranker-4B. The tagged artifact
remains available for reference and rollback.

## Capabilities

The v1 column describes the `v2.0.0a1` artifact; the v1 server has since been
[removed from the source](#v1-removal).

| Capability | v1 server (`v2.0.0a1`) | v2 preview |
|---|---|---|
| Semantic search (keyword backstop) and native reranking | Yes | Yes |
| Immutable captures, indexing, search | Yes | Yes |
| Consolidation and contradiction detection | Yes | Not implemented |
| RAPTOR summary clusters | Yes | Not implemented |
| Transport | Stdio | HTTP MCP, REST, stdio shim |
| File watching and serialized daemon writes | No shared daemon | Yes |
| Index directory | `.lancedb` | `index.lancedb` |

v2 excludes captures from search unless sessions are requested. See the
[v2 guide](V2.md) for scope, authentication, citations, and command details.

## Install an audited artifact

Download the wheel and `SHA256SUMS` from the [releases page](https://github.com/sushiHex/mainframe-mcp/releases)
for the version you intend to install: `v2.0.0b1` is the current build;
`v2.0.0a1` is the historical Qwen build. Verify the wheel hash against that
manifest before installation:

```powershell
Get-FileHash ./mainframe_mcp-2.0.0b1-py3-none-any.whl -Algorithm SHA256
```

On Linux, use `sha256sum -c SHA256SUMS` with both distribution files present.
The manifest detects a mismatched download; it is not a signature.

```bash
python -m venv .venv-preview
# PowerShell: ./.venv-preview/Scripts/Activate.ps1
# POSIX: source .venv-preview/bin/activate
python -m pip install --upgrade pip
# Install CUDA-enabled PyTorch using the selector linked below, then:
python -m pip install ./mainframe_mcp-2.0.0b1-py3-none-any.whl
mainframe --help
```

Choose the appropriate CUDA build using the official [PyTorch installation
selector](https://pytorch.org/get-started/locally/). The CPU-only CI installation
checks do not establish GPU support. Python 3.11 and 3.14 are tested on Windows
and Linux; the declared minimum is 3.11. Hardware presets live in the source
checkout; installed packages also accept an explicit JSON configuration.
Models download on first use unless already cached. Coordinate VRAM capacity
before loading them and check [Windows startup memory](V2.md#startup-memory-diagnostics).

## Migrate deliberately

1. Keep authoritative notes and captures. Install the current public source or
   its audited wheel in the environment used by the daemon and clients.
2. Create a separate v2 configuration using the example in [V2.md](V2.md).
   Select the intended `paths.mainframe_dir` and an explicit `paths.include_projects` list;
   an empty include list selects all eligible projects. Set `service.token` on
   shared machines. Point `MAINFRAME_CONFIG` at this file in every v2 shell/client.
3. Run `mainframe validate --output ~/mainframe-validation/preview-01` before
   production indexing. It uses synthetic files and its own state. For a soak,
   add `--duration 1800`. Keep the generated token, logs, and report private.
4. Start `mainframe serve`, check `mainframe status`, and search the selected
   project. The first start builds a fresh v2 index and re-embeds its corpus;
   it does not reuse or convert v1's `.lancedb`.
5. Register the v2 HTTP MCP endpoint or `mainframe-mcp` after those checks pass.
   Index-setting drift blocks writes; stop the daemon before `mainframe rebuild`.
   `mainframe reload` releases models and resets backoff; restart to read config edits.

For a smaller allocation, use the [shared-GPU profile and scoped trial](V2.md#scoped-project-trial).
Exercise real questions and capture recall through separate MCP sessions before
expanding scope. Continue observing ordinary work over time; synthetic checks
and a bounded trial do not establish retrieval quality or long-term stability.

## v1 removal

The v1 stdio server (`src/mainframe_mcp/`, `python -u -m mainframe_mcp.server`)
was removed from the source after 2.0.0a2. It is not a rollback target: the
`v2.0.0a1` artifact is the last build that shipped it, and its index was
never shared with v2. If one is still on disk:

- `<mainframe_dir>/.lancedb` (v1's index) and `.manifest.json` are never read
  by v2 and can be deleted. v2 keeps `index.lancedb`.
- `research/sessions/` (v1's raw capture lane) stays excluded from the
  knowledge lane; v2 captures live under `<mainframe_dir>/captures/`.
- Configuration keys only v1 read are refused at startup like any other
  unknown key, and the error names the key. Delete these from `config.json`
  if present: `reranker.top_k`, `chunker.min_section_tokens`,
  `search.fetch_multiplier`, `tiers.archive`, `memory.consolidate_threshold`.
- `MAINFRAME_READ_ONLY` and `MAINFRAME_PREWARM` were v1 variables; v2 uses
  the `/mcp/ro` endpoint and `service.prewarm`.

## Maintainer release checks

Build from a clean, reviewed public commit, with output outside the checkout:

```bash
python -m pip install build twine
python -m build --outdir ../mainframe-preview-dist
python -m twine check ../mainframe-preview-dist/*
python scripts/check_install.py --wheel-dir ../mainframe-preview-dist
```

`python -m build` builds the wheel from its generated source distribution.
The source archive includes the complete test harness, presets, and guides;
run `python -m pytest tests/ -q` from its extracted root as well as the checkout.
The installation check verifies every runtime module, both entry points,
configuration, and agreement between runtime and distribution versions.
Require `pytest`, `public-audit`, and the complete `package-install` matrix.
Run live validation separately on supported hardware.

Before uploading, follow [PUBLIC_AUDIT.md](PUBLIC_AUDIT.md): review the exact
commit ancestry, source tree, release text, and every extracted artifact file
and metadata record. Scan archives and extracted contents, inspect the file
inventory, and record hashes in `SHA256SUMS`. Record the build commit, tool
versions, checks, and live-validation scope in the release notes. Keep raw
logs and private corpora outside the publication. Prepare a **draft prerelease**;
publishing release artifacts requires a maintainer decision. Harrier default
promotion on public main is separate from the original tagged artifact.
