import os
import sys

import pytest

from mainframe.core.hashing import content_hash, file_hash
from mainframe.core.paths import canonical, chunk_key, doc_id


def test_canonical_uses_forward_slashes_and_is_absolute(tmp_path):
    p = tmp_path / "docs" / "a.md"
    p.parent.mkdir()
    p.write_text("x", encoding="utf-8")
    c = canonical(str(p))
    assert "\\" not in c and os.path.isabs(c)
    assert canonical(str(p)) == canonical(str(p).replace("/", os.sep))


@pytest.mark.skipif(sys.platform != "win32", reason="case folding is a Windows property")
def test_canonical_folds_case_on_windows(tmp_path):
    p = tmp_path / "Docs" / "A.md"
    p.parent.mkdir()
    p.write_text("x", encoding="utf-8")
    assert canonical(str(p)) == canonical(str(p).lower())


def test_doc_id_is_32_hex_and_stable():
    d = doc_id("c:/repos/x/docs/a.md")
    assert len(d) == 32 and int(d, 16) >= 0
    assert d == doc_id("c:/repos/x/docs/a.md")
    assert d != doc_id("c:/repos/x/docs/b.md")


def test_chunk_key_shape():
    assert chunk_key("ab" * 16, 7) == "ab" * 16 + "#7"


def test_hashes():
    assert file_hash("A b") != file_hash("a b")           # raw bytes matter
    assert content_hash("A   b\n") == content_hash("a b")  # normalized
    assert len(file_hash("")) == 64 and len(content_hash("")) == 64
