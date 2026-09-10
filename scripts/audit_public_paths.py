"""Reject private paths and unreviewed binary content in reachable Git history.

Read-only and dependency-free. Pair with Gitleaks and the manual review in
docs/PUBLIC_AUDIT.md; this check cannot identify confidential ordinary prose.
"""

from pathlib import Path, PurePosixPath
import subprocess
import sys


PRIVATE_ROOTS = ("research/", "eval/results/", "eval/sessions/", "docs/superpowers/",
                 ".claude/", ".codex/", "captures/", ".models/", ".lancedb/", "index.lancedb/")
PRIVATE_FILES = {"eval/results.md", "eval/test_queries.json", "docs/mainframe-mcp-design.md",
                 "config.json", ".manifest.json"}
PRIVATE_SUFFIXES = {".db", ".sqlite", ".sqlite3", ".pem", ".key", ".p12", ".pfx",
                    ".safetensors", ".gguf", ".pt", ".pth"}


def _git(repo: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", *args], cwd=repo)


def _private(path: str) -> bool:
    lowered = path.casefold()
    name = PurePosixPath(lowered).name
    return (lowered.startswith(PRIVATE_ROOTS) or lowered in PRIVATE_FILES
            or name == ".env" or name.startswith(".env.") and name != ".env.example"
            or PurePosixPath(lowered).suffix in PRIVATE_SUFFIXES)


def audit(repo: Path) -> list[str]:
    if _git(repo, "rev-parse", "--is-shallow-repository").strip() == b"true":
        raise ValueError("Fetch full history before auditing (shallow repository refused).")
    commits = _git(repo, "rev-list", "--all").decode("ascii").splitlines()
    if not commits:
        raise ValueError("No committed history to audit; commit locally before publishing.")
    findings, seen, blobs = [], set(), {}
    for commit in commits:
        for entry in filter(None, _git(repo, "ls-tree", "-rz", commit).split(b"\0")):
            meta, raw_path = entry.split(b"\t", 1)
            mode, kind, oid = meta.decode("ascii").split()
            path = raw_path.decode("utf-8")
            key = (path, mode, kind, oid)
            if key in seen:
                continue
            seen.add(key)
            reason = None
            if _private(path):
                reason = "private-by-policy path"
            elif kind != "blob" or mode not in ("100644", "100755"):
                reason = "symlink or submodule requires separate review"
            else:
                if oid not in blobs:
                    data = _git(repo, "cat-file", "blob", oid)
                    try:
                        data.decode("utf-8")
                        blobs[oid] = b"\0" not in data
                    except UnicodeDecodeError:
                        blobs[oid] = False
                if not blobs[oid]:
                    reason = "binary content requires separate review"
            if reason:
                findings.append(f"{commit[:12]} {path!r}: {reason}")
    return findings


def main() -> int:
    try:
        findings = audit(Path.cwd())
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"Public audit could not complete: {exc}", file=sys.stderr)
        return 2
    if findings:
        print("\n".join(findings), file=sys.stderr)
        return 1
    print("Public path/content-type audit passed for all reachable commits.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
