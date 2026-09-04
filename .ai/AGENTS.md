# Diffusers — Agent Guide

## Setup

We recommend developing in a virtual environment managed by [uv](https://docs.astral.sh/uv/):

```bash
uv venv && source .venv/bin/activate
uv pip install -e .                       # provides diffusers-cli
```

List the available skills, and what each one is for, with:

```bash
diffusers-cli skills list
```

Install them with:

```bash
diffusers-cli skills add <skill name>      # install one, or --all for every skill
                                          # --claude / --codex / --cursor to pick one target
```

`diffusers-cli skills update` refreshes what you installed.

Skills used to be installed through the `Makefile`, which symlinked the project-level `.claude/skills` and
`.agents/skills` at `.ai/skills` in this repo. Remove those links if they exist — otherwise the install writes through
them into `.ai/` itself:

```bash
rm -f .claude/skills .agents/skills
```

At the start of a session, confirm the setup and tell the user what you find:

```bash
python utils/check_ai.py                  # guides and skills are consistent
diffusers-cli skills list                 # which skills are installed
```

Claude Code and Codex can also install via plugins

```bash
claude plugin marketplace add huggingface/diffusers
claude plugin install diffusers@diffusers-skills --scope project

codex plugin marketplace add huggingface/diffusers    # then install from the Plugins Directory
```


## Human in the loop

Everything a PR reviewer sees must come directly from a human, or be explicitly approved by one for the exact content. This covers commit messages, code comments and docstrings, PR titles and descriptions, and any PR/issue comment, review, or reply. Ask whether the user prefers that you write a draft or that they prepare it themselves. Keep every draft short and easy to digest — a human has to genuinely read it before it goes out.

- **Don't commit unless the user approved the exact commit message, and never push or open a PR on
  your own.** The user decides when anything is published, each time.
- **Don't post to GitHub directly** — no comments, reviews, or replies. Draft when asked and hand the text to the user.

## Code formatting

- `make style` and `make fix-copies` should be run before opening a PR

## Reference guides

- **Coding style** — see [code_style.md](references/code_style.md) for how code should read, and the `# Copied from` rules.
- **Models** — see [models.md](references/models.md) for model conventions, attention pattern, implementation rules, dependencies, and gotchas. For adding or converting a model, use the [model-integration](skills/model-integration/SKILL.md) skill.
- **Pipelines** — see [pipelines.md](references/pipelines.md) for pipeline conventions, patterns, and gotchas.
- **Modular pipelines** — see [modular.md](references/modular.md) for modular pipeline conventions, patterns, and gotchas.
- **Tests** — see [testing.md](references/testing.md) for test conventions: required test layers, tester mixins, and dummy-component rules.
- **Reporting** — see [reporting.md](references/reporting.md) for how to write bug reports (what a reproduction is) and performance claims (end-to-end numbers first).

## Skills

Task-specific guides live in `.ai/skills/` and are loaded on demand by AI agents. Available skills include:

- [model-integration](skills/model-integration/SKILL.md) (adding/converting pipelines)
- [custom-blocks](skills/custom-blocks/SKILL.md) (packaging a `ModularPipelineBlocks` subclass for the Hub)
- [diffusers-cli](skills/diffusers-cli/SKILL.md) (running pipelines, inspecting schemas, and using the Diffusers CLI)
- [self-review](skills/self-review/SKILL.md) (pre-PR self-review against the project rules)

## Self-review before a PR

Before opening a PR, run self-review against [review-rules.md](references/review-rules.md). The [self-review skill](skills/self-review/SKILL.md) runs this as the same pass the `@claude` CI reviewer uses. Share the final report on the PR (description or comment) — see the skill for details.
