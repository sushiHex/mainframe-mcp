"""Flat YAML-ish frontmatter: `key: value` lines between `---` fences. Every
value is flattened to one line so a newline can never close the block early
or inject keys. No YAML library: the shapes we write are all flat scalars."""

_FENCE = "---\n"


def _scalar(v) -> str:
    return " ".join(str(v).replace("\\", "/").splitlines()).strip()


def render(fields: dict) -> str:
    lines = [f"{k}: {_scalar(v)}" for k, v in fields.items()]
    return _FENCE + "\n".join(lines) + "\n" + _FENCE


def split(text: str):
    """(fields, body). fields is {} when there is no frontmatter block."""
    if not text.startswith(_FENCE):
        return {}, text
    end = text.find("\n" + _FENCE, len(_FENCE))
    if end == -1:
        return {}, text
    block = text[len(_FENCE):end]
    body = text[end + 1 + len(_FENCE):]
    fields = {}
    for line in block.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            fields[k.strip()] = v.strip()
    return fields, body
