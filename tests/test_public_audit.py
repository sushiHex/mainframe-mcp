"""The publication check reads Git history, including deleted private files."""

import importlib.util
from pathlib import Path
import subprocess

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_public_paths.py"


def _git(repo, *args):
    return subprocess.check_output(["git", "-c", "user.name=Audit fixture",
                                    "-c", "user.email=audit@example.invalid",
                                    "-c", "commit.gpgsign=false", *args], cwd=repo)


def _commit(repo, path, data=b"synthetic fixture\n"):
    target = repo / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    _git(repo, "add", "--", path)
    _git(repo, "commit", "-qm", "Fixture: add synthetic content")


@pytest.fixture
def audit_repo(tmp_path):
    _git(tmp_path, "init", "-q")
    _commit(tmp_path, "README.md")
    spec = importlib.util.spec_from_file_location("public_audit", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return tmp_path, module.audit


def test_public_text_and_templates_are_allowed(audit_repo):
    repo, audit = audit_repo
    _commit(repo, "eval/results.template.md")
    _commit(repo, "tests/test_capture.py")
    assert audit(repo) == []


@pytest.mark.parametrize("path", ["research/notes.md", "eval/results.md",
                                   "eval/test_queries.json", "eval/sessions/run.md",
                                   "docs/superpowers/plan.md", "config.json",
                                   ".env", "nested/.env.local", "state/events.db"])
def test_private_paths_block_publication(audit_repo, path):
    repo, audit = audit_repo
    _commit(repo, path)
    assert any(path in finding for finding in audit(repo))


def test_deleted_private_file_remains_a_publication_failure(audit_repo):
    repo, audit = audit_repo
    _commit(repo, "research/notes.md")
    _git(repo, "rm", "--", "research/notes.md")
    _git(repo, "commit", "-qm", "Fixture: remove synthetic content")
    assert any("research/notes.md" in finding for finding in audit(repo))


def test_unreviewed_binary_blocks_publication(audit_repo):
    repo, audit = audit_repo
    _commit(repo, "docs/sample.bin", b"\x00synthetic binary")
    assert any("docs/sample.bin" in finding and "binary" in finding for finding in audit(repo))


def test_restoring_regular_file_does_not_hide_historical_symlink(audit_repo):
    repo, audit = audit_repo
    _commit(repo, "link", b"target")
    oid = _git(repo, "rev-parse", "HEAD:link").decode().strip()
    for mode in ("120000", "100644"):
        _git(repo, "update-index", "--cacheinfo", f"{mode},{oid},link")
        _git(repo, "commit", "-qm", "Fixture: change entry mode")
    assert any("symlink" in finding for finding in audit(repo))


def test_shallow_history_is_refused(audit_repo):
    repo, audit = audit_repo
    target = repo / "shallow"
    _git(repo, "clone", "--quiet", "--depth=1", repo.as_uri(), str(target))
    with pytest.raises(ValueError, match="shallow"):
        audit(target)
