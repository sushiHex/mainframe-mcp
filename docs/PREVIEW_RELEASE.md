# 2.0.0a1 Preview

This distribution includes the existing v1 MCP server and the opt-in v2 daemon.
It does not change MCP registrations or migrate an existing index automatically.
Use a separate environment and state directory for evaluation.

## Capabilities

| Capability | v1 server | v2 preview |
|---|---|---|
| Hybrid search and native reranking | Yes | Yes |
| Immutable captures, indexing, search | Yes | Yes |
| Consolidation and contradiction detection | Yes | Not implemented |
| RAPTOR summary clusters | Yes | Not implemented |
| Transport | Stdio | HTTP MCP, REST, stdio shim |
| File watching and serialized daemon writes | No shared daemon | Yes |
| Index directory | `.lancedb` | `index.lancedb` |

v2 excludes captures from search unless sessions are requested. See the
[v2 guide](V2.md) for scope, authentication, citations, and command details.

## Install an audited artifact

Download the wheel and `SHA256SUMS` from the maintainer's preview release when
available. Verify the wheel hash against that manifest before installation:

```powershell
Get-FileHash ./mainframe_mcp-2.0.0a1-py3-none-any.whl -Algorithm SHA256
```

On Linux, use `sha256sum -c SHA256SUMS` with both distribution files present.
The manifest detects a mismatched download; it is not a signature.

```bash
python -m venv .venv-preview
# PowerShell: ./.venv-preview/Scripts/Activate.ps1
# POSIX: source .venv-preview/bin/activate
python -m pip install --upgrade pip
# Install CUDA-enabled PyTorch using the selector linked below, then:
python -m pip install ./mainframe_mcp-2.0.0a1-py3-none-any.whl
mainframe --help
```

Choose the appropriate CUDA build using the official [PyTorch installation
selector](https://pytorch.org/get-started/locally/). The CPU-only CI installation
checks do not establish GPU support. Python 3.10 and 3.14 are tested on Windows
and Linux; the declared minimum is 3.10. Hardware presets live in the source
checkout; installed packages also accept an explicit JSON configuration.
Models download on first use unless already cached. Coordinate VRAM capacity
before loading them and check [Windows startup memory](V2.md#startup-memory-diagnostics).

## Migrate deliberately

1. Keep the current v1 environment, configuration, and MCP registration available.
   Back up configuration and authoritative notes/captures before experimenting.
2. Create a separate v2 configuration using the example in [V2.md](V2.md).
   Set a new `paths.mainframe_dir` and an explicit `paths.include_projects` list;
   an empty include list selects all eligible projects. Set `service.token` on
   shared machines. Point `MAINFRAME_CONFIG` at this file in every v2 shell/client.
3. Run `mainframe validate --output ~/mainframe-validation/preview-01` before
   production indexing. It uses synthetic files and its own state. For a soak,
   add `--duration 1800`. Keep the generated token, logs, and report private.
4. Start `mainframe serve`, check `mainframe status`, and search the selected
   project. The first start builds a fresh v2 index and re-embeds its corpus;
   it does not reuse or convert v1's `.lancedb`.
5. Register the v2 HTTP MCP endpoint or stdio shim only after those checks pass.
   Index-setting drift blocks writes; stop the daemon before `mainframe rebuild`.
   `mainframe reload` releases models and resets backoff; restart to read config edits.

## Roll back

Run `mainframe down` with the v2 configuration and confirm it has stopped. Restore
the original v1 environment, configuration, and MCP registration
(`python -u -m mainframe_mcp.server`). Its separate index remains available.
Keep v2 captures and source notes: rollback does not import them into v1 or undo
changes made to authoritative files. Preserve the preview directory for diagnosis;
do not delete or repoint either index as part of rollback.

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
publishing it or changing defaults is a separate maintainer decision.
