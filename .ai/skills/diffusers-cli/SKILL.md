---
name: diffusers-cli
description: >
  Use when the user wants to run a diffusers pipeline from a terminal (one-off
  generation, batch jobs, smoke-testing a new model), submit jobs to HF Jobs
  hardware via `--remote`, or introspect an unknown pipeline's input schema
  before calling it. Prefer this over writing ad-hoc Python scripts for
  generation tasks.
---

## Overview

`diffusers-cli` is the shipped CLI in `src/diffusers/commands/`. Three subcommands
matter for agentic use:

| Command | Purpose |
| --- | --- |
| `generate` | Run any `DiffusionPipeline` or `ModularPipeline` by forwarding `--pipeline-kwargs` verbatim. Saves output by sniffing its runtime type. |
| `describe` | Print the input schema (kwarg names + types + defaults + docstring) for a pipeline repo. **No weights downloaded** — only `model_index.json` (or `modular_model_index.json`) is fetched. |
| `custom_blocks` | Package a local `ModularPipelineBlocks` subclass for the Hub. |

`env` (system info) and `fp16_safetensors` (deprecated) also exist but aren't
relevant to inference.

## The describe → generate flow

For any model you haven't called before, run `describe` first to learn its
input contract, then `generate` with the right `--pipeline-kwargs`:

```bash
# 1. Discover what kwargs the pipeline takes (no weight download)
diffusers-cli describe --model black-forest-labs/FLUX.1-dev --json

# 2. Run it
diffusers-cli generate \
    --model black-forest-labs/FLUX.1-dev \
    --pipeline-kwargs '{"prompt": "a cat", "num_inference_steps": 30}' \
    --dtype bf16
```

`describe`'s `--json` output is machine-readable: a list of `{name, type_hint,
default, required, description}` entries. Use `--verbose` to additionally parse
the `__call__` docstring's `Args:` block for descriptions on standard pipelines.

## Standard vs modular detection

`generate` auto-detects which kind of pipeline it's calling:

1. If `model_index.json` exists on the repo → `DiffusionPipeline.from_pretrained` path
2. Otherwise → `ModularPipeline.from_pretrained` path

You don't need to tell it which. Modular repos must pass `--trust-remote-code`
if they ship custom block code.

## `--pipeline-kwargs` semantics

A JSON object passed straight through to `pipeline(**kwargs)`. String values at
known image-input keys (`image`, `mask_image`, `control_image`,
`ip_adapter_image`, `image_2`) are auto-loaded as PIL images, so you can pass
URLs or local paths directly:

```bash
diffusers-cli generate \
    --model stabilityai/stable-diffusion-xl-refiner-1.0 \
    --pipeline-kwargs '{
        "image": "https://example.com/cat.png",
        "prompt": "a photorealistic cat",
        "strength": 0.6
    }'
```

**Shell-quoting gotcha**: the JSON must be on one line (or use `\` to
line-continue). A literal newline inside the single-quoted argument lands as a
raw control char inside the string and breaks `json.loads`.

## Output handling

`generate` sniffs the pipeline return type and saves accordingly:

- `PIL.Image` / list of them → `outputs/generate-<i>.png`
- Frame sequence (≥2 PILs or ndarrays) → `outputs/generate-0.mp4` (uses `--fps`, default 8)
- Numpy audio array → `outputs/generate-0.wav` (uses `--sampling-rate`)
- Anything else → JSON dump

Override the destination with `--output <path>` (file or directory).

Use `--push-to <user>/<bucket>` to upload outputs to an HF bucket after saving.
The bucket is created if it doesn't exist; objects land under
`<run_id>/<filename>`.

## Remote execution (`--remote`)

Adds `--remote` to submit the same call as a Hugging Face Job:

```bash
diffusers-cli generate \
    --model black-forest-labs/FLUX.1-dev \
    --pipeline-kwargs '{"prompt": "a cat"}' \
    --remote --flavor a100-large
```

What happens:

1. Token is read from `args.token` or `huggingface_hub.get_token()`.
2. A bucket (`<user>/jobs-artifacts` by default) is auto-created.
3. Job is submitted to HF Jobs via `run_job` with the pytorch image
   (`pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime`) so torch + CUDA are
   preinstalled.
4. Container runs `uv pip install --system --break-system-packages
   <small-deps> && diffusers-cli generate ...` — only ~50 MB of deps install
   because torch already lives in the image's site-packages.
5. The CLI streams the container's logs to stderr until the job terminates,
   then downloads any files the job uploaded to the bucket under its `run_id`
   prefix.
6. A timing breakdown (`queued_seconds`, `run_seconds`, `total_seconds`) is
   printed and added to the JSON payload.

Use `--no-wait` to submit and immediately return the job id without streaming
logs. Use `--namespace` to run under a different account.

## `--json` machine-readable mode

All subcommands accept `--json` to emit a single JSON object on stdout instead
of human-readable text. Use this when an agent needs to parse the result —
output paths, timing, pushed-bucket URIs, etc.

## When NOT to use this skill

- Multi-stage workflows where you need intermediate tensor manipulation between
  pipelines → write Python.
- Training or fine-tuning → CLI only covers inference.
- Anything requiring custom `device_map`, `quantization_config`, or other
  low-level loader knobs not exposed by the CLI flags → write Python.

## Verifying the CLI is installed

The console entry point lives in `pyproject.toml` (`diffusers-cli =
"diffusers.commands.diffusers_cli:main"`). If `diffusers-cli` is not on PATH
after `pip install -e .`, reinstall with `pip install -e . --force-reinstall
--no-deps` and check `which diffusers-cli`.
