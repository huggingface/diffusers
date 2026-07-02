# `diffusers-cli run` — reference

Full surface for `diffusers-cli run`. Use this file as the source of truth when constructing a `run`
invocation. The top-level [`SKILL.md`](SKILL.md) covers when to use the CLI; this file covers how.

## The schema → run flow

For any model you haven't called before, run `schema` first to learn its input contract, then `run` with
the right `--pipeline-kwargs`:

```bash
# 1. Discover what kwargs the pipeline takes (no weight download)
diffusers-cli --format json schema --model black-forest-labs/FLUX.2-klein-9B

# 2. Run it
diffusers-cli run \
    --model black-forest-labs/FLUX.2-klein-9B \
    --pipeline-kwargs '{"prompt": "Make the cats fur grey", "image": "https://blobcdn.same.energy/a/d0/58/d058b51c2329b0ea4057e9f12cd9a1da36347e34"}' \
    --dtype bf16
```

`schema --format json` emits a `{task, model, pipeline_class, inputs[]}` payload where each input is
`{name, type_hint, default, required, description}`.

## Standard vs modular detection

`run` auto-detects which kind of pipeline it's calling:

1. If `model_index.json` exists on the repo → `DiffusionPipeline.from_pretrained` path.
2. Otherwise → `ModularPipeline.from_pretrained` path.

You don't need to tell it which. Modular repos must pass `--trust-remote-code` if they ship custom block code.

## `--pipeline-kwargs` semantics

A JSON object passed straight through to `pipeline(**kwargs)`. String values at known image-input keys (`image`,
`mask_image`, `control_image`, `ip_adapter_image`, `image_2`) are auto-loaded as PIL images, so you can pass URLs
or local paths directly:

```bash
diffusers-cli run \
    --model black-forest-labs/FLUX.2-klein-9B \
    --pipeline-kwargs '{"image": "https://example.com/cat.png", "prompt": "make the fur grey", "strength": 0.6}'
```

**Shell-quoting gotcha**: the JSON must be on one line (or use `\` to line-continue). A literal newline inside the
single-quoted argument lands as a raw control char inside the string and breaks `json.loads`.

## LoRA adapters (`--lora`)

Attach a LoRA after the pipeline loads via a JSON spec:

```bash
diffusers-cli run \
    --model black-forest-labs/FLUX.2-klein-9B \
    --pipeline-kwargs '{"prompt": "a tiny grey cat"}' \
    --lora '{"lora_id": "alvdansen/littletinies", "lora_scale": 0.8}'
```

Calls `pipeline.load_lora_weights(<lora_id>, adapter_name="default")` and, if `lora_scale` is present,
`pipeline.set_adapters(["default"], adapter_weights=[<scale>])`. Errors clearly if the pipeline doesn't support
LoRA or `lora_id` is missing.

## Optimization flags

- `--dtype {auto, bf16, fp16, fp32, …}` — pipeline weight dtype. `bf16` is the right default for modern DiTs on
  A100/H100.
- `--cpu-offload {model, group}` — `model` uses `enable_model_cpu_offload`, `group` uses
  `enable_group_offload(offload_type="leaf_level", use_stream=True)`. Use `group` to fit a 9B+ model on a single A100.
- `--attention-backend {default, flash_hub, flash_varlen_hub, flash_4_hub, sage_hub}` — hub-hosted kernels,
  auto-downloaded on first use. Failures (kernel not available, CUDA arch mismatch, network) raise a clear
  `SystemExit` listing the alternatives instead of silently reverting to the default.
- `--vae-tiling` / `--vae-slicing` — lower peak VAE decode VRAM.
- `--context-parallel` — Ulysses-style context parallelism on a DiT. See [Context parallel](#context-parallel) below.

`disable_mmap=True` is always passed to `from_pretrained` — sequential reads are faster than mmap page-faults on
most filesystems.

## Output handling

`run` sniffs the pipeline return type and saves accordingly:

- `PIL.Image` / list of them → `outputs/run-<i>.png`
- Frame sequence (≥2 PILs or ndarrays) → `outputs/run-0.mp4` (uses `--fps`, default 8)
- Numpy audio array → `outputs/run-0.wav` (uses `--sampling-rate`)
- Anything else → JSON dump

Override the destination with `--output <path>` (file or directory).

Use `--push-to <user>/<bucket>` to upload outputs to an HF bucket after saving. The bucket is created if it
doesn't exist; objects land under `<run_id>/<filename>`.

## Remote execution (`--remote`)

Adds `--remote` to submit the same call as a Hugging Face Job:

```bash
diffusers-cli run \
    --model black-forest-labs/FLUX.2-klein-9B \
    --pipeline-kwargs '{"prompt": "Make the cats fur grey", "image": "https://blobcdn.same.energy/a/d0/58/d058b51c2329b0ea4057e9f12cd9a1da36347e34"}' \
    --remote --flavor a100-large \
    --dtype bf16 \
    --cpu-offload group
```

What happens:

1. Your HF token is picked up (from `--token` or your login).
2. A bucket (`<user>/jobs-artifacts` by default) is created if it doesn't exist.
3. The job runs in a pytorch container that already has torch + CUDA preinstalled. Only the small Python
   deps (`diffusers`, `accelerate`, `transformers`, `safetensors`) are installed at container start — about
   50 MB instead of 3 GB.
4. Container logs stream to your terminal. When the job finishes, the CLI downloads every file the job
   uploaded to the bucket under its `run_id` prefix into `./outputs/`.
5. A timing breakdown (`queued_seconds`, `run_seconds`, `total_seconds`) is printed and included in the JSON
   payload.

Flags:

- `--flavor <name>` — HF Jobs hardware (e.g. `a10g-small`, `a100-large`, `4xa100-large`).
- `--timeout <duration>` — max wallclock (e.g. `30m`, `2h`). Defaults to `10m`.
- `--dependencies <pkg>` — extra pip deps (repeatable).
- `--namespace <name>` — run under a different account.
- `--no-wait` — submit, return job id, don't stream logs.
- `--push-to <bucket>` — override the artifact bucket id.

## Context parallel

`--context-parallel` enables Ulysses CP on a DiT-based pipeline. **Locally** the user must launch via torchrun:

```bash
torchrun --nproc-per-node=2 -m diffusers.commands.diffusers_cli run \
    --model black-forest-labs/FLUX.2-klein-9B \
    --pipeline-kwargs '{"prompt": "Make the cats fur grey"}' \
    --dtype bf16 \
    --context-parallel
```

**Remotely** the CLI handles the torchrun wrapping — just pass `--context-parallel` to a `--remote` invocation on
a multi-GPU flavor:

```bash
diffusers-cli run \
    --model black-forest-labs/FLUX.2-klein-9B \
    --pipeline-kwargs '{"prompt": "Make the cats fur grey", "image": "https://blobcdn.same.energy/a/d0/58/d058b51c2329b0ea4057e9f12cd9a1da36347e34"}' \
    --remote --flavor 4xa100-large \
    --dtype bf16 \
    --context-parallel
```

Inside the container, CP swaps the entrypoint to `torchrun --nproc-per-node=gpu -m
diffusers.commands.diffusers_cli`, initializes a hybrid process group (`cpu:gloo,cuda:nccl` — NCCL for the
attention all-to-all, Gloo for `ulysses_anything`'s per-rank size coordination), pins each rank to
`cuda:{LOCAL_RANK}`, and gates output saving/printing to rank 0 only.

**Memory note**: CP shards the sequence, **not the weights**. Every rank still holds the full transformer. Wins
are wall-clock attention speedup and headroom for very long sequences, not "fit a model that doesn't fit." For
weight sharding you'd want TP or FSDP — not exposed in the CLI yet.

CP is DiT-only. UNet pipelines raise a clear error directing you to a DiT pipeline (FLUX, SD3, HunyuanDiT,
AuraFlow, …).

## Output mode (`--format`)

The CLI auto-detects when running under an AI coding agent (Claude Code, Cursor, Aider, GH Copilot Agent — via
`CLAUDECODE`, `CLAUDE_CODE`, `CURSOR_AI`, `AIDER_AI_CONTEXT`, `GH_COPILOT_AGENT`) and switches output to **agent
mode** automatically — TSV tables, `key=value` results, compact JSON dicts, no progress bars.

Override explicitly with `--format {auto, human, agent, json}` placed **before** the subcommand:

```bash
diffusers-cli --format json run --model <id> --pipeline-kwargs '...'
```

The legacy `--json` flag on `run` still works as a shortcut for `--format json`.
