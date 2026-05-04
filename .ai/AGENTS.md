# Diffusers — Agent Guide

## Coding style

Strive to write code as simple and explicit as possible.

- Prefer inlining small helper/utility functions over factoring them out — a reader should be able to follow the full flow without jumping between functions. If a private helper has only one caller, inlining it at the call site is usually the cleaner choice.
- No defensive code, unused code paths, or legacy stubs — do not add fallback paths, safety checks, or configuration options "just in case"; do not carry unused method parameters "for API consistency", backwards-compatibility aliases for names that never shipped, or deprecation shims for code that was never released. When porting from a research repo, delete training-time code paths, experimental flags, and ablation branches entirely — only keep the inference path you are actually integrating.
- Do not guess user intent and silently correct behavior. Make the expected inputs clear in the docstring, and raise a concise error for unsupported cases rather than adding complex fallback logic.

Before opening the PR, self-review against [review-rules.md](review-rules.md), which collects the most common mistakes we catch in review.

---

## Code formatting

- `make style` and `make fix-copies` should be run as the final step before opening a PR

### Copied Code

- Many classes are kept in sync with a source via a `# Copied from ...` header comment
- Do not edit a `# Copied from` block directly — run `make fix-copies` to propagate changes from the source
- Remove the header to intentionally break the link

### Models

- See [models.md](models.md) for model conventions, attention pattern, implementation rules, dependencies, and gotchas.
- See the [model-integration](./skills/model-integration/SKILL.md) skill for the full integration workflow, file structure, test setup, and other details.

### Pipelines & Schedulers

- See [pipelines.md](pipelines.md) for pipeline conventions, patterns, and gotchas.

### Modular Pipelines

- See [modular.md](modular.md) for modular pipeline conventions, patterns, and gotchas.

## Skills

Task-specific guides live in `.ai/skills/` and are loaded on demand by AI agents. Available skills include:

- [model-integration](./skills/model-integration/SKILL.md) (adding/converting pipelines)
- [parity-testing](./skills/parity-testing/SKILL.md) (debugging numerical parity).
