<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Command line interface

`diffusers-cli` is a command line entry point for running, inspecting, and packaging Diffusers pipelines
without writing Python. It's installed alongside the `diffusers` package.

The CLI adapts its output for whoever's reading it:

- Under an AI coding agent (Claude Code, Cursor, Aider, GitHub Copilot Agent — detected via env vars like
  `CLAUDECODE`, `CURSOR_AI`, `AIDER_AI_CONTEXT`), output switches to a compact `key=value` / TSV format with
  no progress bars.
- In a normal terminal, it prints human-friendly summaries.
- Force a specific format with `--format {auto, human, agent, json, quiet}`.
- Control log verbosity with the `DIFFUSERS_VERBOSITY` environment variable (`debug`, `info`, `warning`,
  `error`, `critical`; defaults to `warning`). Set to `info` to surface progress messages like sandbox
  creation, dependency install, and file upload counts.

## Available commands

| Command | Purpose |
|---|---|
| [`env`](#env) | Print environment info for bug reports. |
| [`schema`](#schema) | Inspect a pipeline's `__call__` signature without downloading weights. |
| [`run`](#run) | Run a pipeline locally or in a Hugging Face Sandbox. |
| [`custom_blocks`](#custom_blocks) | Package a local `ModularPipelineBlocks` subclass for the Hub. |
| [`fp16_safetensors`](#fp16_safetensors) | Convert a checkpoint to fp16 `.safetensors`. |
| [`skills`](#skills) | Install pre-authored skill bundles into your AI coding agent. |

> [!TIP]
> This page covers the common flags. For the full, always-current list of options for any subcommand, run `diffusers-cli <command> --help` (`diffusers-cli run --help`).

## `env`

Prints Python/PyTorch/Diffusers versions, CUDA info, and installed optional deps. Use it when opening an
issue so maintainers can reproduce your setup.

```bash
diffusers-cli env
```

## `schema`

`schema` reads `model_index.json` for standard pipelines, `modular_model_index.json` for modular pipelines, or
`modular_config.json` for custom block repositories. Standard pipelines expose their `__call__` signature, and
modular repositories expose declared block inputs. For modular repos with custom code, pass
`--trust-remote-code`.

```bash
diffusers-cli --format json schema --model black-forest-labs/FLUX.1-dev
```

## `run`

Run a pipeline end-to-end. Auto-detects standard vs modular repos, auto-loads media inputs from URLs or local
paths, saves outputs by detecting the pipeline's return type, and can submit the same call to a Hugging Face
Sandbox via `--remote`.

Minimal example:

```bash
diffusers-cli run \
    --model black-forest-labs/FLUX.1-dev \
    --dtype bf16 \
    --pipeline-kwargs '{"prompt": "an astronaut riding a horse"}'
```

### Passing pipeline arguments

`--pipeline-kwargs` takes a JSON object that's forwarded to `pipeline(**kwargs)`. String values at known
media-input keys are auto-loaded:

- Images (`image`, `mask_image`, `control_image`, `ip_adapter_image`, `image_2`) → `PIL.Image` via
  `load_image`.
- Videos (`video`, `control_video`) → `list[PIL.Image]` via `load_video`.
- Audio (`initial_audio_waveforms`, `reference_audio`, `src_audio`) → `torch.Tensor` via `torchaudio.load`.

```bash
diffusers-cli run \
    --model black-forest-labs/FLUX.2-klein-9B --dtype bf16 \
    --pipeline-kwargs '{"prompt": "make the fur grey", "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png", "strength": 0.6}'
```

Both media keys and text keys accept a JSON array to run a batch through a single pipeline call. Each entry
in a media list is loaded individually (URL, local path, or bucket-mount path), and diffusers processes the
whole list in one forward pass on the GPU:

```bash
diffusers-cli run \
    --model black-forest-labs/FLUX.1-Kontext-dev --dtype bf16 \
    --pipeline-kwargs '{
        "prompt": ["make it grey", "make it pink", "make it blue"],
        "image": [
            "https://.../cat1.png",
            "https://.../cat2.png",
            "https://.../cat3.png"
        ]
    }'
```

### Loading

Configure how the CLI loads model weights and custom pipeline code.

- `--dtype {auto, bfloat16, bf16, float16, fp16, float32, fp32}` — weight dtype.
- `--device-map <value>` — component placement. Accepts a torch device string (`cuda`, `cuda:0`, `cpu`, `mps`),
  `balanced` (auto-splits components across visible GPUs), or a JSON dict for explicit per-component placement.
  Auto-detected if omitted. See [device_map](../training/distributed_inference#device_map) for more details
- `--variant fp16` — pick a weight variant.
- `--revision <sha>` — pin a specific model revision.
- `--trust-remote-code` — allow custom code from the Hub (required for repos that ship custom pipeline classes
  or modular blocks). See [Community pipelines](../using-diffusers/custom_pipeline_overview) for standard custom
  pipelines and [Modular Diffusers](../modular_diffusers/overview).
- `--lora <spec>` — attach one or more LoRA adapters after loading. Accepts a single JSON object
  or a list of them. `lora_id` is required per entry; `lora_scale` defaults to `1.0`; `adapter_name`
  is optional (auto-generated as `lora_<i>` when stacking).
  - Single: `--lora '{"lora_id": "alvdansen/flux-koda", "lora_scale": 0.8}'`
  - Multiple: `--lora '[{"lora_id": "alvdansen/flux-koda", "lora_scale": 0.6, "adapter_name": "koda"}, {"lora_id": "Shakker-Labs/FLUX.1-dev-LoRA-AntiBlur", "lora_scale": 0.4}]'`

  All specs are loaded by `pipeline.load_lora_weights(...)`, then activated together with a single
  `pipeline.set_adapters(names, adapter_weights=scales)` call. See [LoRA](../tutorials/using_peft_for_inference)
  for a deeper walkthrough of adapter stacking, scale scheduling, and hotswapping.

### Optimizations

- `--cpu-offload {model, group}` — `model` calls `enable_model_cpu_offload`; `group` calls
  `enable_group_offload(offload_type="leaf_level", use_stream=True)`. Onload target comes from `--device-map`
  (which must be a plain device string for offload). See
  [Model offloading](../optimization/memory#model-offloading) and
  [Group offloading](../optimization/memory#group-offloading).
- `--attention-backend {default, flash_hub, flash_varlen_hub, flash_4_hub, sage_hub}` — Hub-hosted attention
  kernels, auto-downloaded on first use. Transformer-based pipelines only; ignored with a warning on legacy UNet
  pipelines. See [Attention backends](../optimization/attention_backends).
- `--vae-tiling` / `--vae-slicing` — lower VAE decode VRAM. See
  [VAE tiling](../optimization/memory#vae-tiling) and [VAE slicing](../optimization/memory#vae-slicing).
- `--compile [JSON]` — compile denoiser modules with [torch.compile](../optimization/fp16#torchcompile). The
  CLI prefers [regional compilation](../optimization/fp16#regional-compilation) for modules with repeated
  blocks. Bare `--compile` uses `fullgraph=true`. A JSON object is forwarded to `torch.compile`. Not supported
  with `--context-parallel`.
- `--context-parallel` — Ulysses-style context parallelism on a DiT-based pipeline. Locally requires torchrun;
  under `--remote` the CLI wraps `torchrun --nproc-per-node=gpu` for you. See
  [Context parallelism](../training/distributed_inference#context-parallelism).

### Outputs

`run` detects the pipeline output type:

- `PIL.Image`/list → `<task>-<i>.png`
- Frame sequence → `<task>-0.mp4` (`--fps` controls framerate, default 8)
- Audio array → `<task>-0.wav` (`--sampling-rate` controls rate)
- Anything else → JSON dump

Default output directory is `~/.diffusers/cli/run/outputs/diffusers-run-<YYYYMMDDTHHMMSS>-<uuid>/`. Each run
gets its own subdirectory so consecutive invocations don't overwrite. Override with `--output <path>` (file or
directory) — explicit paths bypass the namespaced default.

Use `--push-to <your-bucket-id>` to upload outputs to a
[Hugging Face storage bucket](https://huggingface.co/docs/hub/en/storage-buckets). `<your-bucket-id>` is a
bare bucket id in `<namespace>/<name>` form (same shape as a Hub repo id) — `hf://` URIs are not accepted here.
The bucket is created if missing; objects land under `<run_id>/<filename>` and are addressable as
`hf://buckets/<your-bucket-id>/<run_id>/<filename>`.

The table below describes remote runs. For local runs, `--push-to` uploads the locally saved output. It does
not suppress local file creation.

| `--push-to` set? | `--output` set? | Result |
|---|---|---|
| no | no | download to default local dir |
| no | yes | download to `--output` |
| yes | no | bucket only, no local download |
| yes | yes | bucket AND `--output` |

`--format` shapes the stdout metadata (paths, timing, sandbox info) — it does not change the file format of
the media itself. Written images are always PNG, videos MP4, audio WAV.

### Remote execution (`--remote`)

Run the same call inside a [Hugging Face Sandbox](https://huggingface.co/docs/huggingface_hub/en/guides/sandbox)
— an isolated cloud VM the CLI drives over HTTP: it uploads inputs, installs deps, runs the pipeline, downloads
outputs, then terminates the sandbox. Requires `huggingface_hub>=1.23`.

```bash
diffusers-cli run \
    --model black-forest-labs/FLUX.1-dev --dtype bf16 \
    --pipeline-kwargs '{"prompt": "an astronaut riding a horse"}' \
    --remote --flavor a100-large
```

Remote flags:

- `--flavor <name>` — sandbox hardware (e.g. `a10g-small`, `h200`, `rtx-pro-6000`).
- `--timeout <duration>` — max wallclock for the run command inside the sandbox (default `10m`).
- `--dependencies <pkg>` — extra pip deps (repeatable). Useful for pinning a diffusers branch tarball or
  adding pipeline-specific extras.
- `--namespace <name>` — create the sandbox under a different HF org/account.
- `--image <ref>` — override the sandbox image. Must ship torch + CUDA compatible with your `--flavor`'s
  driver.
- `--volume <bucket-id>[:<mount-path>]` — mount an [HF storage bucket](https://huggingface.co/docs/hub/en/storage-buckets)
  into the sandbox as a read-write directory. Repeatable. Default mount path is
  `/mnt/buckets/<bucket-id>`. Reference mounted files from `--pipeline-kwargs` like any other local path.
  Applied only on new sandbox creation — ignored when reconnecting via `--sandbox-id`.

By default each `--remote` run is ephemeral (create → run → download → kill). To reuse a warm sandbox across
runs — keeping deps, the model weight cache, and the `torch.compile` cache on its disk — keep it alive and
reconnect:

- `--keep-alive` — don't terminate the sandbox after the run; its id is printed.
- `--sandbox-id <id>` — reconnect to a kept-alive sandbox instead of creating a new one. Implies `--keep-alive`.
- `--idle-timeout <duration>` — auto-shutdown after this much inactivity (defers to the Sandbox default of
  `10m` when unset); the billing backstop. Pass `none` to disable the auto-reaper; the sandbox then runs
  until the 24h hard cap unless killed. Applied only on new sandbox creation — ignored when reconnecting via
  `--sandbox-id`.

```bash
# First run keeps the sandbox alive and prints sandbox_id=<id>.
diffusers-cli run -m black-forest-labs/FLUX.1-dev --dtype bf16 \
    --pipeline-kwargs '{"prompt": "a cat"}' --remote --flavor a100-large --keep-alive

# Reconnect for the next run — the model is already cached, so only inference runs.
diffusers-cli run -m black-forest-labs/FLUX.1-dev --dtype bf16 \
    --pipeline-kwargs '{"prompt": "a dog"}' --remote --flavor a100-large --sandbox-id <id>

# Stop it when done (or let --idle-timeout reap it).
hf sandbox kill <id>
```

#### Batch editing with mounted buckets

Combine `--volume` (inputs), `--keep-alive` (warm sandbox across runs), and `--push-to` (outputs) to run a
pipeline over a batch of inputs sitting in an HF bucket:

```bash
# One-time: upload your source images to a bucket.
hf buckets create alice/edit-inputs
hf buckets upload alice/edit-inputs ./local-cats/ cats/

# Warm the sandbox once — mount the input bucket, load the model, keep it alive.
diffusers-cli run \
    -m black-forest-labs/FLUX.1-Kontext-dev --dtype bf16 \
    --remote --flavor a100-large --keep-alive \
    --volume alice/edit-inputs \
    --push-to alice/edit-outputs \
    --pipeline-kwargs '{"prompt": "make it grey", "image": "/mnt/buckets/alice/edit-inputs/cats/01.png"}'
# Prints sandbox_id=<id>. Note it.

# Loop the rest of the batch through the warm sandbox — each call reuses the loaded weights.
for img in $(hf buckets ls alice/edit-inputs cats | awk '{print $NF}'); do
    diffusers-cli run \
        -m black-forest-labs/FLUX.1-Kontext-dev --dtype bf16 \
        --remote --sandbox-id <id> \
        --push-to alice/edit-outputs \
        --pipeline-kwargs "{\"prompt\": \"make it grey\", \"image\": \"/mnt/buckets/alice/edit-inputs/cats/$img\"}"
done

hf sandbox kill <id>
```

The container FUSE-mounts the bucket at `/mnt/buckets/alice/edit-inputs/`, so `image:` in
`--pipeline-kwargs` resolves to a real file inside the sandbox and no local upload happens. Outputs land in
`alice/edit-outputs` via `--push-to` — each run gets its own `<run_id>/` prefix so files don't overwrite.

## `custom_blocks`

Package a local `ModularPipelineBlocks` subclass for upload to the Hub. Reads a Python file, AST-scans it for
subclasses of `ModularPipelineBlocks`, instantiates the chosen one, and calls `save_pretrained` in the current
working directory.

```bash
# Package the first block found in ./block.py
diffusers-cli custom_blocks

# Point at a different file / pick a specific class
diffusers-cli custom_blocks --block_module_name my_block.py --block_class_name MyDenoiseBlock
```

The block class must be instantiable with zero constructor args — hardcode defaults in `__init__` or read
config from the pipeline `state` at call time.

## `fp16_safetensors`

Convert a checkpoint on the Hub to fp16 `.safetensors` and push the result. Useful for shrinking a repo's
weight size for faster loading. See `diffusers-cli fp16_safetensors --help` for the exact args.

## `skills`

Install Agent Skills bundles from the diffusers repo (`.ai/skills/`).

```bash
# Install the diffusers-cli skill
diffusers-cli skills add diffusers-cli

# Install every skill in the registry
diffusers-cli skills add --all

# List available skills
diffusers-cli skills list

# Preview a skill's SKILL.md without installing
diffusers-cli skills preview diffusers-cli

# Refetch and reinstall every managed skill
diffusers-cli skills update

# Install to the user-level directory instead of the current project
diffusers-cli skills add diffusers-cli --global
```

`--force` overwrites an existing install. `--all` fetches every skill in the registry in one call
and downgrades individual failures to warnings so one broken skill doesn't abort the batch.
