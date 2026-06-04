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

## Dead code analysis (new models)

When reviewing a PR that adds a new model, trace how the model is actually called from the pipeline to identify likely dead code. Include the results as a **suggestions / additional info** section in your review (not as blocking comments — the findings are advisory).

1. **Trace the call path.** Read the pipeline's `__call__` and follow every call into the model — which arguments are passed, which branches are taken, which helper methods are invoked.
2. **Check the default model config.** Look at the default config values in the model's `__init__` (or any published config JSON). Identify code paths that are unreachable under those defaults — e.g. an `if self.config.use_foo:` branch where `use_foo` defaults to `False` and no published checkpoint sets it to `True`.
3. **Flag unused parameters and methods.** Parameters declared in `forward` (or helper methods) but never passed by the pipeline, private methods never called, layers initialized but never used in `forward`.
4. **Qualify findings.** The actual model config can differ from the defaults, so any dead code identified this way is *likely* dead — not certain. Frame findings accordingly: "Under the default config and the pipeline's call path, this code appears unreachable." The PR author may know of configs or use cases that exercise the path.
