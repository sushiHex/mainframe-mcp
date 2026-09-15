import json
import os
from pathlib import Path

import pytest

from mainframe.core.paths import canonical
from mainframe.memory.frontmatter import render
from mainframe.memory.lanes import (ADHOC_PROJECT, Lanes, capture_id_of, load_receipts,
                                    save_receipts)
from v2.helpers import write_md


def _repo(cfg, name):
    return Path(cfg["paths"]["repos_dir"]) / name


def test_knowledge_scanner_matches_v1_rules(cfg):
    r = _repo(cfg, "proj")
    write_md(r / "CLAUDE.md", "# c")
    write_md(r / "AGENTS.md", "# a")
    write_md(r / "SOUL.md", "# identity")               # never indexed
    write_md(r / "README.md", "# readme")               # SKIP_FILES
    write_md(r / "docs" / "arch.md", "# arch")
    write_md(r / "research" / "memory" / "proj.md", "# note")
    write_md(r / "research" / "sessions" / "old.md", "# v1 raw lane — kept separate from v2 captures")
    write_md(r / "node_modules" / "docs" / "x.md", "# skipped dir")
    # Scratch trees hold whole copies of a project's docs/: one real corpus had 14,666 of
    # its 14,934 indexed documents under tmp/, ~60 duplicates of each real file.
    write_md(r / "tmp" / "t137" / "red" / "docs" / "arch.md", "# scratch copy of a real doc")
    write_md(r / "src" / "notes.md", "# not in a knowledge dir")
    (_repo(cfg, ".hidden")).mkdir()
    files = Lanes(cfg).knowledge_files()
    names = sorted(Path(f.path).name for f in files)
    # canonical() lower-cases on Windows, so accept either spelling
    assert names == ["agents.md", "arch.md", "claude.md", "proj.md"] or \
           names == ["AGENTS.md", "CLAUDE.md", "arch.md", "proj.md"]
    assert {f.lane for f in files} == {"knowledge"} and {f.project for f in files} == {"proj"}


def test_project_scope_include_and_exclude(cfg):
    write_md(_repo(cfg, "a") / "docs" / "x.md", "# a")
    write_md(_repo(cfg, "b-tmp") / "docs" / "x.md", "# b")
    write_md(_repo(cfg, "c") / "docs" / "x.md", "# c")

    cfg["paths"]["exclude_projects"] = ["*-tmp"]
    L = Lanes(cfg)
    files = L.knowledge_files()
    assert {f.project for f in files} == {"a", "c"}
    assert L.skipped_projects == 1
    assert L.in_scope("a") and L.in_scope("c") and not L.in_scope("b-tmp")

    cfg["paths"]["include_projects"] = ["a"]
    cfg["paths"]["exclude_projects"] = []
    L2 = Lanes(cfg)
    files2 = L2.knowledge_files()
    assert {f.project for f in files2} == {"a"}
    assert L2.skipped_projects == 2
    assert L2.in_scope("a") and not L2.in_scope("b-tmp") and not L2.in_scope("c")


def test_capture_lane_layout_and_exclusions(cfg):
    L = Lanes(cfg)
    d = L.capture_dir("proj")
    write_md(d / "2026-01-01-000000-ab12cd34-ef56ab78.md", render({"capture_id": "ef56ab78"}) + "# S\n")
    write_md(L.quarantine_dir("proj") / "junk.md", "# junk")
    write_md(L.overflow_dir("proj") / "over.md", "# over")
    write_md(L.capture_dir(ADHOC_PROJECT) / "2026-01-01-000001-x-y.md", "# adhoc")
    files = L.capture_files()
    assert sorted(Path(f.path).name for f in files) == ["2026-01-01-000000-ab12cd34-ef56ab78.md", "2026-01-01-000001-x-y.md"]
    assert {f.lane for f in files} == {"capture"}
    assert {f.project for f in files} == {"proj", ADHOC_PROJECT}
    assert [f.project for f in L.capture_files("proj")] == ["proj"]


def test_capture_id_from_frontmatter_or_body(tmp_path):
    a = write_md(tmp_path / "a.md", render({"capture_id": "deadbeefdeadbeef"}) + "# S\nbody\n")
    b = write_md(tmp_path / "b.md", "# S\nbody\n")
    assert capture_id_of(a) == "deadbeefdeadbeef"
    assert len(capture_id_of(b)) == 16 and capture_id_of(b) == capture_id_of(write_md(tmp_path / "c.md", "# S\nbody\n"))


def test_capture_id_resolves_from_head_without_a_full_read(tmp_path, monkeypatch):
    # capture_id is in the frontmatter (first few lines); the body is > 100 KB.
    # Path.read_text is monkeypatched to blow up, proving capture_id_of never
    # needs to read past the 4 KB head to resolve it.
    big = write_md(tmp_path / "big.md",
                   render({"capture_id": "deadbeefdeadbeef"}) + "# S\n" + ("x" * (100 * 1024 + 1)) + "\n")

    def _boom(self, *a, **kw):
        raise AssertionError("capture_id_of should not need a full read here")
    monkeypatch.setattr(Path, "read_text", _boom)
    assert capture_id_of(big) == "deadbeefdeadbeef"
    monkeypatch.undo()

    # legacy (no frontmatter, so no capture_id in the head) path still falls
    # back to a full read + hash of the body, exactly as before this change.
    legacy = write_md(tmp_path / "legacy.md", "# S\nbody\n")
    assert len(capture_id_of(legacy)) == 16 and capture_id_of(legacy) == \
        capture_id_of(write_md(tmp_path / "legacy2.md", "# S\nbody\n"))


def test_receipts_roundtrip_and_pending(cfg):
    L = Lanes(cfg)
    d = L.capture_dir("proj")
    c1 = write_md(d / "2026-01-01-000000-s1-c1.md", render({"capture_id": "c1"}) + "# one\n")
    c2 = write_md(d / "2026-01-02-000000-s2-c2.md", render({"capture_id": "c2"}) + "# two\n")
    assert [Path(f.path).name for f in L.pending("proj")] == [c1.name, c2.name]
    save_receipts(L.note_receipts_path("proj", "proj"), {"c1"}, {"generated_at": "t"})
    assert load_receipts(L.note_receipts_path("proj", "proj")) == {"c1"}
    assert [Path(f.path).name for f in L.pending("proj")] == [c2.name]
    # draft receipts never define pending
    save_receipts(L.draft_receipts_path("proj", "proj"), {"c2"}, {})
    assert [Path(f.path).name for f in L.pending("proj")] == [c2.name]
    assert json.loads(L.note_receipts_path("proj", "proj").read_text())["capture_ids"] == ["c1"]


def test_load_receipts_tolerates_non_object_json(tmp_path):
    p = write_md(tmp_path / "bad.json", "[1, 2, 3]")
    assert load_receipts(p) == set()
    p2 = write_md(tmp_path / "bad2.json", "\"just a string\"")
    assert load_receipts(p2) == set()


def test_paths_are_inside_lanes(cfg):
    L = Lanes(cfg)
    assert str(L.note_path("proj", "proj")).replace("\\", "/").endswith("/repos/proj/research/memory/proj.md")
    assert str(L.draft_path("proj", "proj")).replace("\\", "/").endswith("/mainframe/drafts/proj/proj.md")
    assert str(L.capture_dir("proj")).replace("\\", "/").endswith("/mainframe/captures/proj")


def test_unsafe_project_or_slug_is_rejected(cfg):
    L = Lanes(cfg)
    for bad in ("../evil", "C:/evil", "a/b", ".hidden", "", "_other"):
        with pytest.raises(ValueError):
            L.capture_dir(bad)
        with pytest.raises(ValueError):
            L.note_path("proj", bad)
    assert L.capture_dir(ADHOC_PROJECT).name == ADHOC_PROJECT   # _adhoc stays valid


def test_capture_files_validates_explicit_project(cfg):
    L = Lanes(cfg)
    for bad in ("../evil", "C:/evil", "a/b", ".hidden", ""):
        with pytest.raises(ValueError):
            L.capture_files(bad)


def test_pending_skips_unreadable_capture(cfg):
    L = Lanes(cfg)
    d = L.capture_dir("proj")
    good = write_md(d / "2026-01-01-000000-s1-c1.md", render({"capture_id": "c1"}) + "# one\n")
    (d / "2026-01-02-000000-s2-c2.md").mkdir()   # a directory named like a capture: unreadable as a file
    assert [Path(f.path).name for f in L.pending("proj")] == [good.name]


def test_one_rule_for_enumeration_and_resolve(cfg):
    L = Lanes(cfg)
    r = _repo(cfg, "Proj")
    write_md(r / "Docs" / "arch.md", "# arch")                                # mixed-case knowledge dir
    write_md(r / "CLAUDE.md", "# c")
    write_md(r / "docs" / "README.md", "# skip")                              # SKIP_FILES, any case
    write_md(r / "Node_Modules" / "docs" / "x.md", "# skip dir, any case")
    write_md(r / "research" / "archive" / "research" / "sessions" / "x.md", "# nested v1 lane")
    write_md(r / "docs" / "note.MD", "# upper-case extension")
    (r / "docs" / "dir.md").mkdir()                                            # a DIRECTORY named like a doc
    rr = _repo(cfg, "research")                                                # a PROJECT named research
    write_md(rr / "sessions" / "docs" / "s.md", "# project named research: sessions/ is not the v1 lane")
    write_md(rr / "research" / "sessions" / "t.md", "# but research/sessions inside it still is")
    enumerated = {f.path: f for f in L.knowledge_files()}
    every = [p for p in Path(cfg["paths"]["repos_dir"]).rglob("*") if p.is_file()]
    resolved = {}
    for p in every:
        lf = L.resolve(p)
        if lf is not None:
            resolved[lf.path] = lf
    assert resolved == enumerated                                              # THE property
    # Name the FILES and let `canonical()` say how they are spelled. Hard-coding
    # the lower-cased forms pinned this to Windows, where normcase folds case;
    # on POSIX `CLAUDE.md` and `note.MD` keep theirs and the suite went red on
    # CI for a difference that is not about lane membership at all.
    expected = {Path(canonical(p)).name for p in (r / "Docs" / "arch.md", r / "CLAUDE.md",
                                                  r / "docs" / "note.MD",
                                                  rr / "sessions" / "docs" / "s.md")}
    assert {Path(k).name for k in enumerated} == expected
    assert {f.project for f in enumerated.values()} == {"Proj", "research"}


def test_linked_project_dirs_are_not_enumerated_twice(cfg):
    L = Lanes(cfg)
    real = _repo(cfg, "Real")
    doc = write_md(real / "docs" / "a.md", "# a")
    link = Path(cfg["paths"]["repos_dir"]) / "Link"
    try:
        import _winapi
        _winapi.CreateJunction(str(real), str(link))
    except (ImportError, AttributeError, OSError):
        try:
            os.symlink(real, link, target_is_directory=True)
        except (OSError, NotImplementedError):
            pytest.skip("cannot create a directory link here")
    files = L.knowledge_files()
    assert [f.path for f in files] == [canonical(doc)] and files[0].project == "Real"
    assert L.resolve(link / "docs" / "a.md") == files[0]


def _link_dir(target: Path, link: Path):
    """A directory link, by whatever mechanism this machine allows. A Windows
    junction needs no privilege, which is exactly why it is the realistic
    containment threat here."""
    link.parent.mkdir(parents=True, exist_ok=True)
    try:
        import _winapi
        _winapi.CreateJunction(str(target), str(link))
        return
    except (ImportError, AttributeError, OSError):
        pass
    try:
        os.symlink(target, link, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("cannot create a directory link here")


def test_a_link_out_of_the_project_is_not_enumerated(cfg, tmp_path):
    """`knowledge_files()` stored a link's RESOLVED target with no
    containment test while `resolve()` realpaths first and rejects it — so a
    full rescan indexed what the watcher refuses, and `docs/linked -> C:/secrets`
    put arbitrary readable files in the index. os.walk follows a junction, so
    the walk reaches the target and only the canonical path shows it left."""
    outside = tmp_path / "outside"
    write_md(outside / "secret.md", "# secret\n")
    r = _repo(cfg, "proj")
    inside = write_md(r / "docs" / "ok.md", "# ok\n")
    _link_dir(outside, r / "docs" / "linked")

    L = Lanes(cfg)
    assert [f.path for f in L.knowledge_files()] == [canonical(inside)]
    assert L.resolve(r / "docs" / "linked" / "secret.md") is None
    assert L.resolve(outside / "secret.md") is None


def test_a_link_into_a_non_knowledge_dir_is_not_a_knowledge_file(cfg):
    """Containment closed the escape but not the CLASSIFICATION: with
    `docs/vendor -> src`, `_classify` ran on the link's spelling (`docs/...`)
    while the stored path was `src/...`, which `resolve()` rejects. `rescan`
    then found the document in both `todo` and `vanished` and upserted and
    deleted it forever, calling `optimize()` each round.

    The property, stated directly: the enumerators admit exactly what
    `resolve()` admits."""
    L = Lanes(cfg)
    r = _repo(cfg, "proj")
    write_md(r / "src" / "x.md", "# not knowledge\n")
    write_md(r / "docs" / "real.md", "# knowledge\n")
    _link_dir(r / "src", r / "docs" / "vendor")

    enumerated = {f.path for f in L.knowledge_files()}
    resolvable = {canonical(p) for p in Path(cfg["paths"]["repos_dir"]).rglob("*")
                  if p.is_file() and L.resolve(p) is not None}
    assert enumerated == resolvable
    assert enumerated == {canonical(r / "docs" / "real.md")}


def test_a_link_within_the_project_enumerates_the_file_once(cfg):
    """Containment is about leaving the root, not about links: a link that
    stays inside still yields its target, exactly once — the canonical path is
    the document's identity, and both spellings resolve to it."""
    r = _repo(cfg, "proj")
    note = write_md(r / "research" / "note.md", "# note\n")
    _link_dir(r / "research", r / "docs" / "linked")

    files = Lanes(cfg).knowledge_files()
    assert [f.path for f in files] == [canonical(note)]


def test_a_capture_link_out_of_the_lane_is_not_enumerated(cfg, tmp_path):
    """The capture lane is flat, so only a FILE link can leave it — which needs
    a privilege Windows does not grant by default. `_within` below carries the
    unconditional coverage of the shared predicate."""
    L = Lanes(cfg)
    outside = write_md(tmp_path / "outside" / "secret.md", "# secret\n")
    real = write_md(L.capture_dir("proj") / "2026-01-01-000000-s-c.md", "# cap\n")
    link = L.capture_dir("proj") / "2026-01-01-000001-s-d.md"
    try:
        os.symlink(outside, link)
    except (OSError, NotImplementedError):
        pytest.skip("cannot create a file link here")

    assert [f.path for f in L.capture_files()] == [canonical(real)]
    assert L.resolve(link) is None


def test_within_is_a_prefix_test_on_whole_segments(cfg):
    """The containment predicate both enumerators share. Whole segments, or
    `repos/proj-old` would count as inside `repos/proj`."""
    W = Lanes(cfg)._within
    assert W("c:/repos/proj", "c:/repos/proj") is True
    assert W("c:/repos/proj/", "c:/repos/proj/docs/a.md") is True
    assert W("c:/repos/proj", "c:/repos/proj-old/docs/a.md") is False
    assert W("c:/repos/proj", "c:/repos/other/docs/a.md") is False
    assert W("c:/repos/proj", "c:/secrets/x.md") is False
    # Folding case and separators is `canonical()`'s job, not `_within`'s —
    # `_within` compares strings that are already canonical. Windows normcase
    # makes these one path; on POSIX they are genuinely two, so asserting True
    # unconditionally was asserting the platform rather than the contract.
    folded = os.path.normcase("C:\\repos\\Proj").replace("\\", "/")
    assert W(folded, "c:/repos/proj/docs/a.md") is (os.name == "nt")


def test_resolve_capture_lane_and_scope(cfg):
    L = Lanes(cfg)
    cap = write_md(L.capture_dir("proj") / "2026-01-01-000000-s-c.md", "# cap")
    q = write_md(L.quarantine_dir("proj") / "x.md", "# quarantined")
    assert L.resolve(cap) == next(f for f in L.capture_files("proj"))
    assert L.resolve(q) is None
    cfg["paths"]["exclude_projects"] = ["proj"]
    assert Lanes(cfg).resolve(write_md(_repo(cfg, "proj") / "docs" / "a.md", "# a")) is None


def test_watch_roots_and_ensure_dirs(cfg):
    L = Lanes(cfg)
    assert not (Path(cfg["paths"]["mainframe_dir"]) / "captures").exists()
    L.ensure_dirs()
    roots = L.watch_roots()
    assert Path(cfg["paths"]["repos_dir"]) in roots and Path(cfg["paths"]["mainframe_dir"]) / "captures" in roots


def test_watch_roots_narrow_to_in_scope_projects(cfg):
    """With an allowlist, watching all of repos_dir means every out-of-scope
    repo's build/git churn lands on the single watchdog dispatcher thread.
    Watch the in-scope project dirs instead, and bound the volume at the source."""
    write_md(_repo(cfg, "a") / "docs" / "x.md", "# a")
    write_md(_repo(cfg, "b") / "docs" / "x.md", "# b")
    cfg["paths"]["include_projects"] = ["a"]
    L = Lanes(cfg)
    L.ensure_dirs()
    roots = L.watch_roots()
    assert _repo(cfg, "a") in roots
    assert _repo(cfg, "b") not in roots and Path(cfg["paths"]["repos_dir"]) not in roots
    assert Path(cfg["paths"]["mainframe_dir"]) / "captures" in roots


def test_could_belong_is_a_pure_string_predicate(cfg):
    """The watcher's FILE relevance test. It must NOT stat: a delete (and the
    source half of a rename) is dispatched after the file is gone, and
    `resolve`'s is_file() check dropped exactly those events — so removals
    depended on a full-lane rescan. Over-approximating is safe: the pipeline
    resolves authoritatively and no-ops a non-member."""
    L = Lanes(cfg)
    r = _repo(cfg, "proj")
    gone = r / "docs" / "a.md"                       # deliberately never created
    assert L.could_belong(gone) is True
    assert L.could_belong(str(gone).replace("/", "\\")) is True      # either separator spelling
    assert L.could_belong(r / "CLAUDE.md") is True
    assert L.could_belong(r / "docs" / "deep" / "b.MD") is True
    assert L.could_belong(r / "docs" / "README.md") is False         # SKIP_FILES
    assert L.could_belong(r / "node_modules" / "docs" / "x.md") is False
    assert L.could_belong(r / "research" / "sessions" / "s.md") is False
    assert L.could_belong(r / "src" / "notes.md") is False           # not a knowledge dir
    assert L.could_belong(r / "docs" / "a.txt") is False
    assert L.could_belong(r / "SOUL.md") is False
    assert L.could_belong(Path(cfg["paths"]["repos_dir"]) / "loose.md") is False
    assert L.could_belong(Path(cfg["paths"]["repos_dir"]) / ".hidden" / "docs" / "x.md") is False
    assert L.could_belong(L.capture_dir("proj") / "2026-01-01-000000-s-c.md") is True
    assert L.could_belong(L.quarantine_dir("proj") / "x.md") is False
    assert L.could_belong(Path(cfg["paths"]["mainframe_dir"]) / ".lancedb" / "x.md") is False
    assert L.could_belong(Path(cfg["paths"]["mainframe_dir"]).parent / "elsewhere" / "docs" / "a.md") is False
    cfg["paths"]["exclude_projects"] = ["proj"]
    assert Lanes(cfg).could_belong(gone) is False


def test_could_belong_admits_exactly_what_resolve_admits(cfg):
    """The over-approximation property, checked against the authoritative rule:
    every file `resolve` accepts, `could_belong` accepts too."""
    L = Lanes(cfg)
    r = _repo(cfg, "Proj")
    write_md(r / "Docs" / "arch.md", "# arch")
    write_md(r / "CLAUDE.md", "# c")
    write_md(r / "docs" / "README.md", "# skip")
    write_md(r / "Node_Modules" / "docs" / "x.md", "# skip")
    write_md(r / "research" / "sessions" / "x.md", "# v1 raw lane")
    write_md(L.capture_dir("proj") / "2026-01-01-000000-s-c.md", "# cap")
    for p in Path(cfg["paths"]["repos_dir"]).rglob("*"):
        if p.is_file() and L.resolve(p) is not None:
            assert L.could_belong(p) is True, p
    for f in L.capture_files():
        assert L.could_belong(f.path) is True


def test_dir_matters_bounds_the_rescan_trigger(cfg):
    """every directory event under ~/repos used to ask for a full
    reconcile, so an out-of-scope repo's `.git/objects` or `node_modules` churn
    drove a rescan every tick. `mainframe_dir` is ALWAYS out: its own
    index.lancedb / events.db / daemon.log writes must never feed the watcher."""
    repos, mf = Path(cfg["paths"]["repos_dir"]), Path(cfg["paths"]["mainframe_dir"])
    cfg["paths"]["exclude_projects"] = ["nope"]
    L = Lanes(cfg)
    assert L.dir_matters(repos) is True                                   # a new project may have appeared
    assert L.dir_matters(repos / "proj" / "docs" / "new") is True
    assert L.dir_matters(repos / "proj") is True
    # Inside a project, only a KNOWLEDGE-dir subtree can hold lane files, and
    # watchdog emits a DirModified on the parent of every changed file — so
    # `src/`, test fixtures and caches must not each cost a full-lane rescan.
    assert L.dir_matters(repos / "proj" / "src") is False
    assert L.dir_matters(repos / "proj" / "evals" / "fixtures") is False
    assert L.dir_matters(repos / "proj" / ".pytest_cache" / "v") is False
    assert L.dir_matters(repos / "proj" / "packages" / "x" / "docs") is True      # nested, still knowledge
    assert L.dir_matters(repos / "proj" / "research" / "memory") is True
    assert L.dir_matters(repos / "nope" / "docs") is False                # out of scope
    assert L.dir_matters(repos / ".hidden" / "docs") is False
    assert L.dir_matters(repos / "proj" / ".git" / "objects" / "ab") is False
    assert L.dir_matters(repos / "proj" / "node_modules" / "x") is False
    assert L.dir_matters(repos / "proj" / "research" / "sessions") is False
    assert L.dir_matters(repos / "proj" / "research" / "sessions" / "deep") is False
    assert L.dir_matters(mf / "captures") is True
    assert L.dir_matters(mf / "captures" / "proj") is True
    # the capture lane is flat: quarantine/ and overflow/ are neither indexed
    # nor pending, so nothing below <project> can change what a rescan sees
    assert L.dir_matters(mf / "captures" / "proj" / "quarantine") is False
    assert L.dir_matters(mf / "captures" / "proj" / "overflow") is False
    assert L.dir_matters(mf) is False
    assert L.dir_matters(mf / ".lancedb") is False
    assert L.dir_matters(mf / ".lancedb" / "chunks.lance" / "data") is False
    assert L.dir_matters(repos.parent / "elsewhere") is False


def test_dir_matters_excludes_a_mainframe_dir_nested_under_repos(cfg):
    """nesting mainframe_dir inside repos_dir must not make the index's own
    writes flag a rescan — the mainframe_dir test wins over the repos test."""
    cfg["paths"]["mainframe_dir"] = str(Path(cfg["paths"]["repos_dir"]) / "mainframe")
    L = Lanes(cfg)
    mf = Path(cfg["paths"]["mainframe_dir"])
    assert L.dir_matters(mf / ".lancedb") is False
    assert L.dir_matters(mf) is False
    assert L.dir_matters(mf / "captures" / "proj") is True     # except the capture lane itself
    assert L.could_belong(mf / ".lancedb" / "x.md") is False
