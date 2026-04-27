# PR Review Rules

Review-specific rules for Claude. Focus on correctness — style is handled by ruff.

Before reviewing, read and apply the guidelines in:
- [AGENTS.md](AGENTS.md) — coding style, copied code
- [models.md](models.md) — model conventions, attention pattern, implementation rules, dependencies, gotchas
- [pipelines.md](pipelines.md) — pipeline conventions, coding style, gotchas
- [modular.md](modular.md) — modular pipeline conventions, patterns, common mistakes
- [skills/parity-testing/SKILL.md](skills/parity-testing/SKILL.md) — testing rules, comparison utilities
- [skills/parity-testing/pitfalls.md](skills/parity-testing/pitfalls.md) — known pitfalls (dtype mismatches, config assumptions, etc.)

## Common mistakes

Common mistakes are covered in the common-mistakes / gotcha sections in [AGENTS.md](AGENTS.md), [models.md](models.md), [pipelines.md](pipelines.md), and [modular.md](modular.md). Additionally, watch for below patterns that aren't covered there:

- **Ephemeral context.** Comments, docstrings, and files that only made sense to the current PR's author or reviewer don't help a future reader/user/developer. Examples: `# per reviewer comment on PR #NNNN`, `# as discussed in review`, `# TODO from offline chat`, debug printouts. Same for files: parity harnesses, comparison scripts, anything in `scripts/` with hardcoded developer paths or imports from the reference repo. State the *reason* so the comment stands alone, or drop it.
