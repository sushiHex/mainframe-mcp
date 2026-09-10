<!-- Public main is the development branch. Audit every commit before pushing;
     see CONTRIBUTING.md and docs/PUBLIC_AUDIT.md. Keep the diff focused. -->

## What & why

<!-- One paragraph: the problem, the change, the mechanism. -->

## Checklist

- [ ] Public audit completed before pushing; audited commit, commands, and finding dispositions recorded below
- [ ] Tests included, written test-first (for fixes: the regression test fails without the fix — say so)
- [ ] `python -m pytest tests/` green locally; new tests are GPU-free (no model loads, no network, no CUDA assumptions)
- [ ] No secrets, personal paths, or real-looking credentials anywhere in the diff (CI leak-check enforces this)
- [ ] Docs updated if behavior changed (`CLAUDE.md` = agent-facing brief, `README.md` = user-facing)
- [ ] Cross-platform: `pathlib`, no hardcoded separators/drive letters

## Retrieval-affecting? (chunking / embedding / reranking / pool / boosts / dedup)

- [ ] Not retrieval-affecting
- [ ] Retrieval-affecting — described the expected effect below; I have **not** claimed an unmeasured quality win. (Maintainer runs the eval gate before merge; if you measured on your own corpus, include methodology per `eval/results.template.md`.)

<!-- Agents: report outcomes faithfully. "Suite green" only if you ran it. -->

## Public audit

<!-- Record commit/range, scanner and private-path checks, manual review, and
     any narrowly scoped synthetic-fixture exceptions. Never paste secrets. -->
