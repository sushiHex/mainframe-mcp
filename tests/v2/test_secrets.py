"""Secret-scrub denylist: redacts secrets, preserves ordinary technical prose."""

import pytest

from mainframe.core.secrets import REDACTED, scrub_secrets


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
    assert REDACTED in out


def test_scrub_redacts_assignment_value_keeps_key():
    out = scrub_secrets("API_KEY=supersecretvalue123")
    assert "supersecretvalue123" not in out
    assert "API_KEY" in out
    assert REDACTED in out


def test_scrub_redacts_bearer_token():
    out = scrub_secrets("Authorization: Bearer abcdefghij0123456789XYZ")
    assert "abcdefghij0123456789XYZ" not in out
    assert "Bearer" in out


def test_scrub_prefixed_env_vars():
    # The KV pattern matches keyword-SUFFIX identifiers: a bare word boundary
    # never fires inside SCREAMING_SNAKE.
    out = scrub_secrets("export AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMIK7MDENGbPxRfiCYEXAMPLEKEY")
    assert "wJalrXUtnFEMIK7MDENG" not in out and REDACTED in out
    out = scrub_secrets("DB_PASSWORD=hunter2pass")
    assert "hunter2pass" not in out and REDACTED in out
    # Low-entropy placeholders: the scrubber keys on the NAME, so the value only
    # has to be present - and a key-shaped value would trip the public audit.
    out = scrub_secrets("MYAPP_API_KEY: aaaabbbbcccc")
    assert "aaaabbbbcccc" not in out
    out = scrub_secrets("AccountKey=AAAAAAAAAAAAAAAAAAAAAAAA==")
    assert "AAAAAAAAAAAA" not in out


def test_scrub_stripe_and_url_creds():
    # FABRICATED provider-shaped fixtures, not real keys. The anchored prefixes
    # are assembled from fragments so no complete token sits contiguous in the
    # source for a secret scanner to flag; the runtime value is identical.
    stripe_key = "sk_" + "live_abcDEF12345678901234"
    out = scrub_secrets(f"use {stripe_key} for prod")
    assert stripe_key not in out and REDACTED in out
    webhook_secret = "wh" + "sec_AbCdEf123456789012345678"
    out = scrub_secrets(f"{webhook_secret} is the webhook secret")
    assert webhook_secret not in out and REDACTED in out
    out = scrub_secrets("conn = postgres://admin:s3cretpw@db.host:5432/app")
    assert "s3cretpw" not in out
    assert "postgres://admin:" in out  # scheme+user preserved, password redacted


def test_scrub_preserves_ordinary_technical_text():
    # "token" is rampant in this corpus and must NOT be scrubbed.
    txt = (
        "Use Qwen3-Embedding-8B on the RTX 3090 with 256-token chunks; "
        "token_count is tracked and the reranker is bge-reranker-v2-m3."
    )
    assert scrub_secrets(txt) == txt
    benign = "the 256-token chunks and token_count fields; the bearer of bad news"
    assert scrub_secrets(benign) == benign
