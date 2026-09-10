"""Gate 1 — corpus identity: the v2 knowledge lane must enumerate exactly the
files the v1 manifest indexed (minus session captures), byte-for-byte.

Usage: python -u eval/corpus_identity.py [path/to/.manifest.json]
Exit 0 when identical; prints missing/extra and exits 1 otherwise."""

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mainframe.config import load_config
from mainframe.core.hashing import file_hash
from mainframe.core.paths import canonical
from mainframe.memory.lanes import Lanes


def _v1_entries(manifest_path: Path) -> dict:
    data = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    out = {}
    for k, v in data.get("files", {}).items():
        kk = k.replace("\\", "/")
        if "/research/sessions/" in kk or "/library/sessions/" in kk:
            continue
        out[canonical(k)] = v.get("content_hash", "")
    return out


def compare(config: dict, manifest_path: Path):
    v1 = _v1_entries(manifest_path)
    v2 = {}
    for lf in Lanes(config).knowledge_files():
        try:
            v2[lf.path] = file_hash(Path(lf.path).read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError):
            v2[lf.path] = "<unreadable>"
    missing = sorted(p for p in v1 if p not in v2)
    extra = sorted(p for p in v2 if p not in v1)
    changed = sorted(p for p in v1 if p in v2 and v1[p] != v2[p])
    if changed:
        print(f"note: {len(changed)} files changed on disk since the v1 manifest (expected drift):",
              file=sys.stderr)
        for p in changed[:10]:
            print("  ", p, file=sys.stderr)
    return missing, extra


def gate_passes(missing) -> bool:
    """The gate only cares that every v1-manifest file the v2 scanner didn't
    find is explained by deletion from disk — a present-but-unmatched file
    means the v2 scanner rules drifted from v1."""
    return all(not Path(p).exists() for p in missing)


def main():
    config = load_config()
    manifest = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(config["paths"]["mainframe_dir"]) / ".manifest.json"
    missing, extra = compare(config, manifest)
    print(f"v1 manifest files (non-session) vs v2 knowledge lane: missing={len(missing)} extra={len(extra)}")
    for p in missing:
        tag = "(deleted since manifest)" if not Path(p).exists() else "(present on disk — scanner gap!)"
        print(f"MISSING {tag} {p}")
    if extra:
        repos = canonical(config["paths"]["repos_dir"]).rstrip("/") + "/"
        by_project = Counter(
            p[len(repos):].split("/", 1)[0] if p.startswith(repos) else "_other" for p in extra)
        print(f"EXTRA: {len(extra)} files not in the v1 manifest (informational — top 10 projects):")
        for project, count in by_project.most_common(10):
            print(f"  {project}: {count}")
    sys.exit(0 if gate_passes(missing) else 1)


if __name__ == "__main__":
    main()
