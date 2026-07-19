"""Local, dependency-free secret scrubbing for ambient capture.

Session captures are derived from in-context text and (via the SessionEnd hook)
from raw transcripts, which routinely contain pasted tokens, env vars, and keys.
`scrub_secrets` runs a regex denylist over text before it is written to disk.

This is a best-effort denylist, not a guarantee — the raw `research/sessions/`
lane is *also* gitignored as the primary secret-leak defense. Keep this module
free of third-party imports so the standalone `~/.claude/hooks/capture_session.py`
can import it (or inline an identical copy) with zero install.
"""

import re

REDACTED = "[REDACTED]"

# Patterns whose entire match is a secret -> replace the whole match.
_TOKEN_PATTERNS = [
    re.compile(r"ghp_[A-Za-z0-9]{36}"),                 # GitHub personal access token
    re.compile(r"gh[osur]_[A-Za-z0-9]{36}"),            # GitHub oauth/server/user/refresh
    re.compile(r"github_pat_[A-Za-z0-9_]{22,}"),        # GitHub fine-grained PAT
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}"),          # Anthropic API key
    re.compile(r"sk-[A-Za-z0-9]{20,}"),                 # OpenAI-style key
    re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"),        # Slack token
    re.compile(r"AKIA[0-9A-Z]{16}"),                    # AWS access key id
    re.compile(r"AIza[0-9A-Za-z_\-]{35}"),              # Google API key
    re.compile(r"\b(?:sk|rk)_(?:live|test)_[A-Za-z0-9]{16,}"),  # Stripe secret/restricted key
    re.compile(r"\bwhsec_[A-Za-z0-9]{20,}"),            # Stripe webhook secret
    re.compile(r"\bGOCSPX-[A-Za-z0-9_\-]{20,}"),        # GCP OAuth client secret
    re.compile(r"eyJ[A-Za-z0-9_\-]{8,}\.eyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}"),  # JWT
    re.compile(
        # Bounded body so an unterminated "-----BEGIN ... PRIVATE KEY-----"
        # (truncated paste, key-format discussion) can't drive an O(n) DOTALL
        # scan to EOF. Real keys are < 8 KB.
        r"-----BEGIN[ A-Z]*PRIVATE KEY-----[\s\S]{0,8192}?-----END[ A-Z]*PRIVATE KEY-----",
    ),
]

# "Bearer <token>" -> keep the "Bearer " prefix, redact the token. 16+ chars so
# ordinary prose ("bearer of bad news") never trips it.
_BEARER_PATTERN = re.compile(r"(?i)(bearer\s+)([A-Za-z0-9_\-\.=]{16,})")

# key=value / key: value assignments -> keep the key, redact the value.
# Deliberately conservative: an explicit "=" or ":" separator is required, and
# the bare word "token" is NOT a keyword (this corpus is full of "256-token
# chunks", "token_count", etc.). Only assignment-shaped secrets are caught.
# The keyword may be the SUFFIX of an identifier — a plain \b prefix is defeated
# by SCREAMING_SNAKE env vars (underscore is a word char, so `\bsecret` never
# matches in AWS_SECRET_ACCESS_KEY= / DB_PASSWORD=), which are the single most
# common real secret shape.
_KV_PATTERN = re.compile(
    r"(?i)([A-Za-z0-9_-]*(?:api[_-]?key|secret|access[_-]?key|secret[_-]?key|"
    r"client[_-]?secret|account[_-]?key|auth[_-]?token|password|passwd)\b)"
    r"(\s*[:=]\s*)"
    r"['\"]?([^\s'\"]{6,})",  # value: any non-space/non-quote run (covers @!#$ etc.)
)

# scheme://user:password@host -> keep scheme+user, redact only the password.
_URL_CREDS_PATTERN = re.compile(
    r"([a-zA-Z][a-zA-Z0-9+.\-]*://[^/\s:@]{1,64}:)([^@\s]{4,})(@)"
)


# All whole-match token patterns combined into ONE alternation so scrubbing is
# a single pass instead of 10 (order preserved: sk-ant- before sk-, etc.).
_ALL_TOKENS_RE = re.compile("|".join(f"(?:{p.pattern})" for p in _TOKEN_PATTERNS))


def scrub_secrets(text: str) -> str:
    """Return `text` with likely secrets replaced by ``[REDACTED]``."""
    if not text:
        return text
    out = _ALL_TOKENS_RE.sub(REDACTED, text)
    out = _BEARER_PATTERN.sub(lambda m: f"{m.group(1)}{REDACTED}", out)
    out = _KV_PATTERN.sub(lambda m: f"{m.group(1)}{m.group(2)}{REDACTED}", out)
    out = _URL_CREDS_PATTERN.sub(lambda m: f"{m.group(1)}{REDACTED}{m.group(3)}", out)
    return out
