"""Markdown-aware recursive chunking with code block preservation.

Splits on markdown headers first, then recursively splits large sections.
Never splits inside code blocks. 512 tokens target, 25% overlap.
"""

import re
from dataclasses import dataclass

# Approximate tokens as words * 1.3 (conservative for technical content)
TOKENS_PER_WORD = 1.3
DEFAULT_CHUNK_TOKENS = 256   # autoresearch optimized (was 512)
DEFAULT_OVERLAP_RATIO = 0.35  # autoresearch optimized (was 0.25)

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
CODE_FENCE_RE = re.compile(r"^```", re.MULTILINE)


@dataclass
class Chunk:
    text: str
    heading: str  # nearest parent heading
    chunk_index: int
    char_start: int
    char_end: int
    token_count: int


def _estimate_tokens(text: str) -> int:
    return int(len(text.split()) * TOKENS_PER_WORD)


def _find_code_blocks(text: str) -> list[tuple[int, int]]:
    """Return (start, end) char offsets of fenced code blocks."""
    blocks = []
    fences = [m.start() for m in CODE_FENCE_RE.finditer(text)]
    for i in range(0, len(fences) - 1, 2):
        # Find end of closing fence line
        close_end = text.find("\n", fences[i + 1])
        if close_end == -1:
            close_end = len(text)
        blocks.append((fences[i], close_end))
    # An odd fence count means the last block is unclosed — treat it as running
    # to EOF so splits don't fall inside it.
    if len(fences) % 2 == 1:
        blocks.append((fences[-1], len(text)))
    return blocks


def _is_inside_code_block(pos: int, code_blocks: list[tuple[int, int]]) -> bool:
    return any(start <= pos <= end for start, end in code_blocks)


def _split_respecting_code_blocks(
    text: str,
    max_tokens: int,
    overlap_tokens: int,
    code_blocks: list[tuple[int, int]],
    base_offset: int = 0,
) -> list[tuple[str, int, int]]:
    """Split text into chunks, never breaking inside code blocks.

    Returns list of (chunk_text, char_start, char_end) relative to original doc.
    """
    if _estimate_tokens(text) <= max_tokens:
        return [(text, base_offset, base_offset + len(text))]

    chunks = []
    # Split on double newlines first (paragraph boundaries)
    separators = ["\n\n", "\n", ". ", " "]

    for sep in separators:
        parts = text.split(sep)
        if len(parts) <= 1:
            continue

        current = []
        current_start = base_offset

        for part in parts:
            test_text = sep.join(current + [part]) if current else part
            if _estimate_tokens(test_text) > max_tokens and current:
                chunk_text = sep.join(current)
                chunk_end = current_start + len(chunk_text)

                # Check we're not splitting inside a code block. chunk_end is
                # already absolute (current_start starts at base_offset), so do
                # NOT add base_offset again.
                if not _is_inside_code_block(chunk_end, code_blocks):
                    chunks.append((chunk_text, current_start, chunk_end))

                    # Overlap: include last N tokens worth of text
                    overlap_chars = int(overlap_tokens / TOKENS_PER_WORD * 5)  # rough char estimate
                    overlap_start = max(0, len(chunk_text) - overlap_chars)
                    overlap_text = chunk_text[overlap_start:]

                    current = [overlap_text, part]
                    current_start = chunk_end - len(overlap_text)
                else:
                    # Inside code block — keep accumulating
                    current.append(part)
            else:
                current.append(part)

        if current:
            chunk_text = sep.join(current)
            chunks.append((chunk_text, current_start, current_start + len(chunk_text)))

        if chunks:
            break

    # Fallback: if no separator worked, return the whole text as one chunk
    if not chunks:
        chunks = [(text, base_offset, base_offset + len(text))]

    return chunks


def chunk_markdown(
    text: str,
    max_tokens: int = DEFAULT_CHUNK_TOKENS,
    overlap_ratio: float = DEFAULT_OVERLAP_RATIO,
) -> list[Chunk]:
    """Chunk markdown text respecting headers and code blocks.

    Strategy:
    1. Split on markdown headers (each section becomes a candidate chunk)
    2. If a section exceeds max_tokens, recursively split on paragraph boundaries
    3. Never split inside fenced code blocks
    4. Apply overlap between chunks
    """
    if not text.strip():
        return []

    code_blocks = _find_code_blocks(text)
    overlap_tokens = int(max_tokens * overlap_ratio)

    # Find all headings and their positions
    headings = list(HEADING_RE.finditer(text))

    if not headings:
        # No headings — split the whole text
        raw_chunks = _split_respecting_code_blocks(
            text, max_tokens, overlap_tokens, code_blocks
        )
        return [
            Chunk(
                text=t.strip(),
                heading="(no heading)",
                chunk_index=i,
                char_start=s,
                char_end=e,
                token_count=_estimate_tokens(t),
            )
            for i, (t, s, e) in enumerate(raw_chunks)
            if t.strip()
        ]

    # Split into sections by heading
    sections = []
    for i, match in enumerate(headings):
        heading_text = match.group(2).strip()
        section_start = match.start()
        section_end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
        section_text = text[section_start:section_end]
        sections.append((heading_text, section_text, section_start))

    # Include any text before the first heading
    if headings[0].start() > 0:
        preamble = text[: headings[0].start()]
        if preamble.strip():
            sections.insert(0, ("(preamble)", preamble, 0))

    # Merge tiny sections (heading-only, <10 tokens) into the next section
    MIN_SECTION_TOKENS = 10
    merged_sections = []
    pending_prefix = ""
    pending_start = 0
    for i, (heading, section_text, section_start) in enumerate(sections):
        if _estimate_tokens(section_text) < MIN_SECTION_TOKENS and i + 1 < len(sections):
            if not pending_prefix:
                pending_start = section_start  # remember the FIRST tiny section's start
            pending_prefix += section_text + "\n"
            continue
        if pending_prefix:
            section_text = pending_prefix + section_text
            section_start = pending_start
            pending_prefix = ""
        merged_sections.append((heading, section_text, section_start))
    if pending_prefix:
        if merged_sections:
            h, t, s = merged_sections[-1]
            merged_sections[-1] = (h, t + "\n" + pending_prefix, s)
        else:
            merged_sections.append(("(merged)", pending_prefix, pending_start))
    sections = merged_sections

    # Chunk each section
    all_chunks = []
    chunk_idx = 0

    for heading, section_text, section_start in sections:
        if _estimate_tokens(section_text) <= max_tokens:
            if section_text.strip():
                all_chunks.append(
                    Chunk(
                        text=section_text.strip(),
                        heading=heading,
                        chunk_index=chunk_idx,
                        char_start=section_start,
                        char_end=section_start + len(section_text),
                        token_count=_estimate_tokens(section_text),
                    )
                )
                chunk_idx += 1
        else:
            # Section too large — split it
            raw_chunks = _split_respecting_code_blocks(
                section_text, max_tokens, overlap_tokens, code_blocks, section_start
            )
            for t, s, e in raw_chunks:
                if t.strip():
                    all_chunks.append(
                        Chunk(
                            text=t.strip(),
                            heading=heading,
                            chunk_index=chunk_idx,
                            char_start=s,
                            char_end=e,
                            token_count=_estimate_tokens(t),
                        )
                    )
                    chunk_idx += 1

    return all_chunks
