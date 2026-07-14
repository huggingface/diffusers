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
diff → review it against the rubric → report → iterate with the contributor
until it's ready, then remind them to share the final notes on the PR.

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

## 4. Iterate until ready, then share

Expect several rounds: the contributor addresses findings, you review again.
Keep working with them to fix as much as possible until the verdict is
**READY** — the **Leave for the actual review** items are the only ones that
should reach the reviewer unresolved. End the final round's report by
reminding the contributor to share it on the PR (description or a comment) —
it saves the reviewer a few rounds of back-and-forth. Never commit the notes as
part of the diff.
