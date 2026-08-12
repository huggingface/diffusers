# Diffusers — Agent Guide

## Setup

This file is loaded automatically in a checkout, through the root `AGENTS.md` / `CLAUDE.md` symlinks. The
[skills](#skills) are not — install them once, by whichever route fits the agent:

```bash
# Claude Code — installs every skill, namespaced as /diffusers:<name>
claude plugin marketplace add huggingface/diffusers
claude plugin install diffusers@diffusers-skills --scope project

# Codex — register the catalog, then install from the Plugins Directory in the ChatGPT desktop app
codex plugin marketplace add huggingface/diffusers

# Any agent, no marketplace needed
diffusers-cli skills add --all            # or: skills add <name>, and --claude / --codex / --cursor to pick a target
```

Installing a skill copies the [reference guides](#reference-guides) it cites into its own `references/` subdirectory,
so it is self-contained wherever it lands. A skill gets a guide by citing it as `references/<guide>.md`; the guides
themselves live once at `.ai/references/`. `diffusers-cli skills list` shows what is available, and `diffusers-cli
skills update` refreshes what you installed.

When editing the skills in this repo, load them from the working tree instead of installing: `claude --plugin-dir .ai`
for one session, or `make codex` to symlink `.agents/skills` at `.ai/skills` (`make clean-ai` to undo).

## Coding style

Strive to write code as simple and explicit as possible.

- Prefer inlining small helper/utility functions over factoring them out — a reader should be able to follow the full flow without jumping between functions. If a private helper has only one caller, inlining it at the call site is usually the cleaner choice.
- No defensive code, unused code paths, or legacy stubs — do not add fallback paths, safety checks, or configuration options "just in case"; do not carry unused method parameters "for API consistency", backwards-compatibility aliases for names that never shipped, or deprecation shims for code that was never released. When porting from a research repo, delete training-time code paths, experimental flags, and ablation branches entirely — only keep the inference path you are actually integrating.
- Do not guess user intent and silently correct behavior. Make the expected inputs clear in the docstring, and raise a concise error for unsupported cases rather than adding complex fallback logic.

---

## Code formatting

- `make style` and `make fix-copies` should be run before opening a PR

### Copied Code

- Many classes are kept in sync with a source via a `# Copied from ...` header comment
- Do not edit a `# Copied from` block directly — run `make fix-copies` to propagate changes from the source
- Remove the header to intentionally break the link

## Reference guides

- **Models** — see [models.md](references/models.md) for model conventions, attention pattern, implementation rules, dependencies, and gotchas. For adding or converting a model, use the [model-integration](./skills/model-integration/SKILL.md) skill.
- **Pipelines** — see [pipelines.md](references/pipelines.md) for pipeline conventions, patterns, and gotchas.
- **Modular pipelines** — see [modular.md](references/modular.md) for modular pipeline conventions, patterns, and gotchas.
- **Tests** — see [testing.md](references/testing.md) for test conventions: required test layers, tester mixins, and dummy-component rules.

## Skills

Task-specific guides live in `.ai/skills/` and are loaded on demand by AI agents. Available skills include:

- [model-integration](./skills/model-integration/SKILL.md) (adding/converting pipelines)
- [custom-blocks](./skills/custom-blocks/SKILL.md) (packaging a `ModularPipelineBlocks` subclass for the Hub)
- [diffusers-cli](./skills/diffusers-cli/SKILL.md) (running pipelines, inspecting schemas, and using the Diffusers CLI)
- [self-review](./skills/self-review/SKILL.md) (pre-PR self-review against the project rules)

## Self-review before a PR

Before opening a PR, run self-review against [review-rules.md](references/review-rules.md). The [self-review skill](skills/self-review/SKILL.md) runs this as the same pass the `@claude` CI reviewer uses. Share the final report on the PR (description or comment) — see the skill for details.
