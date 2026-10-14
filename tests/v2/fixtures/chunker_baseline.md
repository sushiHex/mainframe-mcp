---
type: note
project: example
tags: [chunking, baseline]
---

# Chunker baseline fixture

This document exists so the chunker can be held to the exact output that the
retrieval baseline was measured under. It mixes the constructs the corpus is
made of: frontmatter, nested headings, tiny sections, long paragraphs that
overflow one chunk, fenced code, tables, and lists.

## Overview

The index is disposable and the files are authoritative. Every chunk records
its heading, its character span in the source file, and its token count, and
the search layer uses those spans to cite line numbers back into the file.

### Tiny section

One line.

### Another tiny section

Also one line, which the chunker may merge with its neighbour.

## A long section

Paragraph one describes the embedding path. Documents are split by heading,
then recursively by paragraph and sentence until each piece fits the token
budget, with a fixed overlap ratio carried between neighbouring pieces so that
a fact straddling a boundary is still retrievable from at least one side. The
overlap is measured in tokens, not characters, and the tokenizer used for the
budget is the same one the embedder uses, which is why changing the embedder
changes the index fingerprint.

Paragraph two describes the reranking path. The candidate pool is fixed at
twenty passages drawn from the vector branch, with the keyword branch acting
as a backstop when the vector branch returns too few candidates. The reranker
scores each query and passage pair with a yes or no judgement, keeping only
the final token's logits, and the top three passages are returned with their
spans so the agent can read the surrounding lines directly from the file.

Paragraph three describes capture. A capture is an immutable markdown file
written once under the capture root, scrubbed for secrets before it reaches
disk, and picked up by the watcher like any other file. Captures are excluded
from search by default, because an accreting lane of session notes would
dilute the fixed candidate pool that curated knowledge competes for, and are
included only when the caller asks for them explicitly.

Paragraph four describes the daemon. One process owns the index, holds a file
lock for its whole life, serialises every mutation through a single writer
thread, and answers health probes from bookkeeping alone so that a probe never
loads a model. A rebuild runs offline under the same lock after the daemon has
been stopped, and the first batch after a rebuild creates the table at
whatever dimension the configured embedder produces.

Paragraph five is deliberately repetitive filler so that this section is long
enough to need more than one chunk and therefore exercises the overlap rule.
Filler sentence number one carries no information. Filler sentence number two
carries no information either. Filler sentence number three restates the first
two. Filler sentence number four restates the third. Filler sentence number
five closes the paragraph without adding anything the reader did not already
know from the sentences that came before it.

## Code and tables

```python
def chunk_markdown(text: str, max_tokens: int = 256, overlap_ratio: float = 0.35):
    """Split by heading, then recursively, carrying overlap between pieces."""
    sections = split_by_heading(text)
    for section in sections:
        yield from split_recursive(section, max_tokens, overlap_ratio)
```

| Setting | Default | Effect |
|---|---|---|
| chunk_size | 256 | Token budget per chunk |
| overlap_ratio | 0.35 | Share of the budget carried into the next chunk |
| strip_frontmatter | false | Keep the YAML header as content |

- First list item, short.
- Second list item, a little longer than the first but still a single line.
- Third list item.
  - A nested item under the third.
  - Another nested item.

## Closing

A final short section, so the document does not end inside a long one.
