---
name: self-review
description: >
  Use before opening a PR, or whenever asked to self-review a diffusers
  contribution. Applies the same rubric as the `@claude` CI (checks the diff
  against .ai/review-rules.md, traces call paths for dead code). Reports findings grouped by
  severity, flagging what to fix before submitting (blocking issues + dead code)
  vs what to leave for the actual review. Report-only — does not edit files.
---

# Self-review

Runs the same rubric as the `@claude` CI reviewer, so you catch issues before a
maintainer does — but over your **whole** PR diff. (The CI scopes itself to
`src/diffusers/` and `.ai/`; for your own PR, also review your tests, docs, and
scripts.) You're already on the branch with the conventions loaded, so: get the
diff → review it against the rubric → report → save the notes to share on the PR.

## 1. Get the diff

```bash
git diff main...HEAD          # use your target branch if not main
```

If the branch trails `main` and the diff looks polluted with unrelated merged
files, scope to your own commits: `git log main..HEAD --oneline`, then
`git show <commit>`.

## 2. Read the rubric

`.ai/review-rules.md` is the canonical rubric (the CI pins it from `main`) — read
it and review against it; don't rely on a remembered copy. For the areas you
touched, also read `.ai/models.md`, `.ai/pipelines.md`, or `.ai/modular.md`.

## 3. Report

- **Blocking issues** — numbered. Each: title → explanation → `file.py:line` →
  impact. Cite the rule, e.g. *Per `.ai/models.md`: "…only keep the inference path."*
- **Non-blocking issues** — same format, lower severity.
- **Dead code (advisory)** — a table: `path:line` · Likely-dead / Used · reason.
- **Summary** — short synthesis and a verdict (**READY** / **NEEDS CHANGES**),
  spelling out:
  - **Fix before submitting** — all blocking issues, and remove the flagged dead code.
  - **Leave for the actual review** — non-blocking issues that aren't obviously
    correct; raise these with the reviewer rather than guessing at them now.

Report only — do not edit files. Be concrete, cite the rule, review the whole
diff, and don't invent issues or flag pure style.

## 4. Share the notes with the reviewer

The report is not just for you — sharing it on the PR is part of the
contribution workflow (see the "AI-assisted and agentic contributions" section
of the contributor guide). Reviewers read it to see what was checked, what was
fixed, and what you deliberately left.

- You may run several rounds (fix findings, re-run). Share the **final
  round's** report — the one that reflects the diff you're actually
  submitting — in the PR description or as a PR comment, including any
  findings you intentionally did not fix and why.
- If you self-review again later (e.g. after addressing reviewer feedback),
  post that round's report as a new PR comment.
- Do not commit the notes as part of the PR diff.
