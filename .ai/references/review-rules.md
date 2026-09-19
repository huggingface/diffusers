# PR Review Rules

Review-specific rules for Claude. Focus on correctness — style is handled by ruff.

Before reviewing, read and apply the guidelines in:
- [code_style.md](code_style.md) — coding style, copied code
- [models.md](models.md) — model conventions, attention pattern, implementation rules, dependencies, gotchas
- [pipelines.md](pipelines.md) — pipeline conventions, coding style, gotchas
- [modular.md](modular.md) — modular pipeline conventions, patterns, common mistakes
- [testing.md](testing.md) — test conventions: required test layers, tester mixins, dummy-component rules. When a PR adds or changes tests, check them against this guide.
- [pitfalls.md](pitfalls.md) — known pitfalls causing numerical discrepancies between the reference implementation and the diffusers port (dtype mismatches, config assumptions, etc.)

## Common mistakes

Common mistakes are covered in the common-mistakes / gotcha sections in [code_style.md](code_style.md), [models.md](models.md), [pipelines.md](pipelines.md), and [modular.md](modular.md). Additionally, watch for below patterns that aren't covered there:

- **Ephemeral context.** Comments, docstrings, and files that only made sense to the current PR's author or reviewer don't help a future reader/user/developer. Examples: `# per reviewer comment on PR #NNNN`, `# as discussed in review`, `# TODO from offline chat`, debug printouts. Same for files: parity harnesses, comparison scripts, anything in `scripts/` with hardcoded developer paths or imports from the reference repo. State the *reason* so the comment stands alone, or drop it.

## Documentation impact

A PR can leave existing docs stale or surface a pattern worth recording. Scan the docs related to what the PR touches and flag updates as a **suggestions / additional info** section (not blocking):

- **Usage docs.** New or changed public behavior — a new pipeline/model, a new argument, changed defaults, a renamed API — should have matching updates in `docs/`, docstrings, and examples. Flag any that now describe outdated behavior or that are missing for the new surface.
- **Agent docs.** If the review turns up a rule, pattern, or common gotcha that isn't written down yet — especially one the author got wrong or that you had to reason out — propose adding it to the relevant agent guide ([code_style.md](code_style.md), [models.md](models.md), [pipelines.md](pipelines.md), [modular.md](modular.md), a skill, or this file) so the next contributor/agent gets it for free instead of repeating the mistake. Human review comments on the PR are a good source for these: if a human reviewer pointed something out and your review missed it, that usually indicates a doc gap — figure out what's missing and propose the addition.

## Dead code analysis (new models)

When reviewing a PR that adds a new model, trace how the model is actually called from the pipeline to identify likely dead code. Include the results as a **suggestions / additional info** section in your review (not as blocking comments — the findings are advisory).

1. **Trace the call path.** Read the pipeline's `__call__` and follow every call into the model — which arguments are passed, which branches are taken, which helper methods are invoked.
2. **Check the released configs.** Search for (or ask the contributor for) the list of released checkpoints, and collect their configs. Remove all the `__init__` config options not used across those releases; the code paths only those options reach go with them (see "Support only what released checkpoints use" in models.md).
3. **Check the runtime arguments.** Same point for `forward` / `__call__` parameters: collect the ones the official examples and recipes never pass, and verify with the contributor that they are actually used — remove the ones that aren't, with the code paths only they reach.
4. **Flag unused parameters, methods, and classes.** Parameters declared in `forward` (or helper methods) but never passed by the pipeline, private methods never called, layers initialized but never used in `forward` — and classes never instantiated: ablation-variant subclasses, intermediate base classes that exist only as inheritance rungs, and aliases giving one class two names.
