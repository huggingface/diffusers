---
name: diffusers-cli
description: >
  Use when the user wants to run a diffusers pipeline from a terminal (one-off
  generation, batch jobs, smoke-testing a new model), submit jobs to HF Jobs
  hardware via `--remote`, introspect a pipeline's input schema before
  calling it, or attach a LoRA at inference time. Prefer this over writing
  ad-hoc Python scripts for generation tasks.
---

## Overview

`diffusers-cli` is the shipped CLI in `src/diffusers/commands/`. Subcommands relevant to agentic use:

| Command         | Purpose                                                                                                                                                                                       |
| --------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `generate`      | Run any `DiffusionPipeline` or `ModularPipeline`. Forwards `--pipeline-kwargs` verbatim, saves output by sniffing its runtime type, optionally runs on HF Jobs via `--remote`.                |
| `describe`      | Print the input schema for a pipeline repo (kwarg names, types, defaults, descriptions). **No weights downloaded** — only the small index file.                                               |
| `custom_blocks` | Package a local `ModularPipelineBlocks` subclass for the Hub.                                                                                                                                 |
| `env`           | Print versions of diffusers + torch + transformers + accelerate + safetensors + CUDA + GPU info. Use when investigating environment issues, dtype/precision support, or building bug reports. |

## When to read which file

Most agentic work goes through `generate`. Read the matching reference file before constructing a command:

- **[`generate.md`](generate.md)** — full reference for `diffusers-cli generate`. Covers `--pipeline-kwargs`
  semantics and the shell-quoting gotcha, LoRA via `--lora`, optimization flags (`--dtype`, `--cpu-offload`,
  `--attention-backend`, `--vae-tiling/slicing`), output handling and `--push-to` bucket uploads, the full
  `--remote` HF Jobs flow (image, container command, log streaming, timing payload, artifact download), and
  context parallel (`--context-parallel`) for both local-torchrun and `--remote` paths.

The other commands are small enough that `diffusers-cli <command> --help` is the canonical reference:

```bash
diffusers-cli describe --help
diffusers-cli custom_blocks --help
diffusers-cli env --help
```

## When NOT to use this skill

- Multi-stage workflows where you need intermediate tensor manipulation between pipelines → write Python.
- Training or fine-tuning → CLI only covers inference.
- Anything requiring custom `device_map`, `quantization_config`, or other low-level loader knobs not exposed by
  the CLI flags → write Python.

## Verifying the CLI is installed

The console entry point is registered in `pyproject.toml` (`diffusers-cli =
"diffusers.commands.diffusers_cli:main"`). If `diffusers-cli` is not on PATH after `pip install -e .`, reinstall
with `pip install -e . --force-reinstall --no-deps` and check `which diffusers-cli`. If the installed binary is
missing recent features (e.g. you see `unrecognized arguments: --lora`), reinstall.

## Output formats

`--format {auto, human, agent, json}` (top-level flag, must appear before the subcommand):

- **`human`** — plain-text indented output for terminals (default when not running under an agent harness). No ANSI color.
- **`agent`** — TSV tables and `key=value` lines. Auto-selected when an agent env var is present
  (`CLAUDECODE`, `CLAUDE_CODE`, `CODEX_SANDBOX`, `CURSOR_AI`, `AIDER_AI_CONTEXT`, `GH_COPILOT_AGENT`,
  `AI_AGENT`). Token-cheap for LLM agents to read.
- **`json`** — compact JSON. Use for programmatic parsing (scripts, services) where type fidelity and nested
  structures matter.

`stdout` carries data; `stderr` carries hints/warnings/progress — parseable output is never polluted.

Rule of thumb: `--format json` for scripts that will `json.loads()` the output, otherwise leave it on
auto-detect (`agent` for LLMs, `human` for terminals).
