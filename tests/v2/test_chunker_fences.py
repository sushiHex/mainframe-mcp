"""A '#' line inside a fenced code block is code (a shell comment), not a
heading (#80). The fence scanner follows CommonMark's rules for what opens and
closes a fence; indented code and HTML comments are out of scope and pinned as
the behavior they already had. GPU-free."""
from mainframe.core.chunker import _find_code_blocks, chunk_markdown

PROSE = "Install the tool by following the numbered steps below, then confirm the version it reports.\n"


def _doc(fence_open: str, fence_close: str | None, body: str = "# install deps\npip install foo\n") -> str:
    closing = f"{fence_close}\n" if fence_close else ""
    return f"# Setup\n\n{PROSE}\n{fence_open}\n{body}{closing}\nAfterwards, run the smoke test.\n"


def _headings(text, **kw):
    return [c.heading for c in chunk_markdown(text, 256, 0.35, **kw)]


def test_a_comment_in_a_bash_fence_keeps_the_enclosing_heading_and_the_block_whole():
    chunks = chunk_markdown(_doc("```bash", "```"), 256, 0.35)
    assert [c.heading for c in chunks] == ["Setup"]
    assert "# install deps\npip install foo" in chunks[0].text and "smoke test" in chunks[0].text


def test_tilde_fences_hide_headings_too():
    assert _headings(_doc("~~~bash", "~~~")) == ["Setup"]


def test_a_longer_closing_fence_closes():
    assert _headings(_doc("```bash", "`````")) == ["Setup"]


def test_a_shorter_fence_does_not_close():
    # The 3-backtick line is code inside the 4-backtick block, so the comment after it is code too.
    body = "# one\n```\n# two\n"
    doc = _doc("````bash", "````", body)
    assert _headings(doc) == ["Setup"]
    assert _find_code_blocks(doc) == [(doc.index("````bash"), doc.rindex("````") + 4)]


def test_the_other_fence_character_does_not_close():
    assert _headings(_doc("```bash", "~~~", "# one\n~~~\n# two\n") + "```\n") == ["Setup"]


def test_a_closing_fence_with_trailing_text_does_not_close():
    doc = "# Setup\n\n" + PROSE + "\n```bash\n# one\n``` still code\n# two\n```\n\nAfterwards, run the smoke test.\n"
    assert _headings(doc) == ["Setup"]


def test_an_unclosed_fence_runs_to_eof():
    doc = _doc("```bash", None) + "# still code\n"
    assert _headings(doc) == ["Setup"]
    assert _find_code_blocks(doc) == [(doc.index("```bash"), len(doc))]


def test_a_heading_after_the_closing_fence_is_a_heading_again():
    doc = _doc("```bash", "```") + "\n# Next\n\nThe next section has enough words in it to stand on its own as a chunk.\n"
    assert _headings(doc) == ["Setup", "Next"]


def test_a_fence_indented_four_spaces_is_not_a_fence():
    """Pin: indented code is out of scope; four spaces is not a fence opener."""
    doc = "# Setup\n\n" + PROSE + "\n    ```bash\n# real heading\n\nA real section with enough words in it to stand alone.\n"
    assert _find_code_blocks(doc) == []
    assert "real heading" in _headings(doc)


def test_an_indented_hash_line_is_not_a_heading():
    """Pin: the column-zero heading regex already ignored indented '#' lines."""
    doc = "# Setup\n\n" + PROSE + "\n    # indented code comment\n\nAfterwards, run the smoke test.\n"
    assert _headings(doc) == ["Setup"]


def test_a_heading_inside_an_html_comment_stays_a_heading():
    """Pin: HTML comments are out of scope; such a heading splits the section as before."""
    doc = "# Setup\n\n" + PROSE + "\n<!--\n# hidden\n\nHidden words that are long enough to make a section of their own.\n-->\n"
    assert "hidden" in _headings(doc)


def test_drop_headings_matching_a_fenced_comment_drops_nothing():
    doc = _doc("```bash", "```")
    assert _headings(doc, drop_headings=[r"^install deps$"]) == ["Setup"]
    assert chunk_markdown(doc, 256, 0.35, drop_headings=[r"^install deps$"]) == chunk_markdown(doc, 256, 0.35)
