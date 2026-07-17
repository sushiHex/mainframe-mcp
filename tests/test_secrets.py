"""Secret-scrub denylist: redacts secrets, preserves ordinary technical prose."""

import pytest

from mainframe_mcp.secrets import scrub_secrets


@pytest.mark.parametrize(
    "token",
    [
        "ghp_" + "B" * 36,
        "sk-ant-api03-" + "x" * 30,
        "AKIA" + "1234567890ABCDEF",
        "xoxb-" + "1234567890-abcdefghij",
    ],
)
def test_scrub_removes_known_tokens(token):
    out = scrub_secrets(f"the leaked value is {token} in the log")
    assert token not in out
    assert "REDACTED" in out


def test_scrub_redacts_assignment_value_keeps_key():
    out = scrub_secrets("API_KEY=supersecretvalue123")
    assert "supersecretvalue123" not in out
    assert "API_KEY" in out
    assert "REDACTED" in out


def test_scrub_redacts_bearer_token():
    out = scrub_secrets("Authorization: Bearer abcdefghij0123456789XYZ")
    assert "abcdefghij0123456789XYZ" not in out
    assert "Bearer" in out


def test_scrub_preserves_ordinary_technical_text():
    # "token" is rampant in this corpus and must NOT be scrubbed.
    txt = (
        "Use Qwen3-Embedding-8B on the RTX 3090 with 256-token chunks; "
        "token_count is tracked and the reranker is bge-reranker-v2-m3."
    )
    assert scrub_secrets(txt) == txt
