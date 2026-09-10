# Audit Before Every Public Push

Public branches, tags, PR descriptions, release assets, and commit messages are
publications. Audit before uploading them. CI runs after publication and is a
second check, not permission to publish first.

## Review the actual publication

1. Confirm `origin` targets the public repository and fetch its current refs.
2. Identify every outgoing commit and tag. Inspect each commit and its complete
   files, including changes later reverted or deleted. A clean final diff does
   not remove sensitive data from intermediate history.
3. Review source, tests, fixtures, documentation, examples, images, metadata,
   generated files, commit messages, and any proposed PR/release text. Look for
   credentials, personal paths, private project names, internal URLs, customer
   data, research notes, captures, corpora, and experiment outputs.
4. Transfer private-repository fixes as reviewed file changes onto public
   history. Never merge private ancestry or force-replace public `main`.

## Automated checks

Use Gitleaks 8.30.1, verifying the downloaded release against its published
SHA256 checksum. Run from the repository root:

```bash
python scripts/audit_public_paths.py
gitleaks git --redact --ignore-gitleaks-allow --max-decode-depth 2 --max-archive-depth 2 --log-opts="--all --full-history" .
git diff --check
```

The path check examines every reachable commit, so adding and then deleting a
private file still fails. It also rejects unreviewed binary files, symlinks,
and submodules. Gitleaks scans tests and documentation as well as source.
These tools do not decide whether ordinary-looking prose is confidential;
manual review is mandatory.

Keep raw reports outside the repository and redact findings in shared output.
Resolve every finding before pushing. For genuinely synthetic fixtures,
document the reason and use an exact commit/path/rule/line fingerprint in
`.gitleaksignore`; never exempt an entire directory. Changes to scanner rules,
exceptions, or CI are themselves part of the audit.

## Record and publish

Record the reviewed commit SHA, outgoing refs, checks run, findings and their
dispositions, and manual-review scope. Verify that the pushed SHA is exactly
the audited SHA. Repeat the audit after any new commit or amendment. Keep
private audit notes outside this repository; only a sanitized summary belongs
in the public PR. Block the push if any finding remains unresolved.

## Baseline reviewed on 2026-09-10

The public refs available locally contained four commits, 67 distinct paths,
and 73 file revisions. Gitleaks reported seven findings representing five
distinct fingerprints in synthetic secret-scrubbing tests, including a
fabricated token in older public history. Those exact fingerprints are
documented in `.gitleaksignore`. Other path/marker matches were placeholder
usernames in tests or literals in the old scanning rules. No confirmed live
credential or private-data file was identified in this scoped audit.

This covers reachable public refs inspected at that time, not deleted GitHub
objects, issues, attachments, or release assets. Any future publication needs
its own audit; the baseline is not a blanket approval.
