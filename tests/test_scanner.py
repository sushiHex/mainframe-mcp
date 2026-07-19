"""scan_projects: prune SKIP_DIRS during traversal, still find nested knowledge."""

from pathlib import Path

from mainframe_mcp.scanner import scan_projects


def test_scan_prunes_skipdirs_but_finds_nested_knowledge(tmp_path):
    repos = tmp_path / "repos"
    proj = repos / "proj"

    # knowledge file nested below a non-knowledge dir -> must still be found
    (proj / "packages" / "x" / "research").mkdir(parents=True)
    (proj / "packages" / "x" / "research" / "deep.md").write_text("# d\nnested research file.\n", encoding="utf-8")
    # top-level research
    (proj / "research").mkdir()
    (proj / "research" / "top.md").write_text("# t\ntop research file.\n", encoding="utf-8")
    # a knowledge file buried inside a SKIP_DIR -> must be pruned
    (proj / "node_modules" / "pkg" / "research").mkdir(parents=True)
    (proj / "node_modules" / "pkg" / "research" / "junk.md").write_text("# j\nvendored junk.\n", encoding="utf-8")
    # a SKIP_FILE inside a knowledge dir -> excluded
    (proj / "research" / "README.md").write_text("# readme\n", encoding="utf-8")

    found = {Path(f["path"]).name for f in scan_projects(repos_dir=repos)}
    assert "deep.md" in found      # nested knowledge dir discovered
    assert "top.md" in found
    assert "junk.md" not in found  # node_modules subtree pruned
    assert "README.md" not in found
