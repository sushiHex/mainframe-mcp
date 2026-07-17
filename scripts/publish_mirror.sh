#!/usr/bin/env bash
# Refresh the public mirror (github.com/sushiHex/mainframe-mcp) from master.
#
# The public repo is a squashed single-commit snapshot of the sanitized tree.
# Development history stays in the private repo (origin). PRIVATE-BY-POLICY
# paths below are excluded even though they are tracked in master — research
# and raw experiment logs never publish, regardless of content audits.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

# Everything tracked in master that must NOT ship publicly:
PRIVATE_PATHS=(
  research                      # ALL research stays private (owner policy)
  eval/results.md               # real experiment log (corpus-specific) — results.template.md ships instead
  eval/sessions                 # internal autoresearch session handoffs
  docs/mainframe-mcp-design.md  # internal design/research artifact
)

test -z "$(git status --porcelain)" || { echo "working tree not clean"; exit 1; }
git branch -D public 2>/dev/null || true
git checkout --orphan public
git rm -rq --cached .
git add -A
for p in "${PRIVATE_PATHS[@]}"; do git rm -rq --cached "$p" 2>/dev/null || true; done
git commit -q -m "Public release snapshot

Squashed snapshot of the sanitized tree; development history and
research stay in the private repository."
COUNT=$(git ls-files | wc -l)
# Pattern-carriers excluded from their own hunt: this script and the CI
# leak-check job both contain the literal pattern.
LEAKS=$(git grep -lE "kfaim|prepped|nbatopshot|Red.?[Pp]ill" -- . ':!scripts/publish_mirror.sh' ':!.github/workflows/tests.yml' | wc -l || true)
git checkout -f master
echo "public snapshot: ${COUNT} files, leak-grep hits: ${LEAKS}"
[ "${LEAKS}" = "0" ] || { echo "LEAK CHECK FAILED — not pushing"; exit 1; }
git push --force public public:main
echo "mirror refreshed: https://github.com/sushiHex/mainframe-mcp"
