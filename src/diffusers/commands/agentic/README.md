# Agentic CLI for Diffusers

Single-command access to common Diffusers use-cases. Designed for AI agents
and humans who need to run image/video/audio generation **without writing
Python scripts**.

Every command below is reachable as `diffusers-cli <command>`. Run
`diffusers-cli <command> --help` for full option documentation.

## How it works

The module integrates with the main CLI through a single function call in
`diffusers_cli.py` — removing it disables everything with no side effects.

```
src/diffusers/commands/agentic/
├── app.py          # register_agentic_commands(subparsers) — single integration point
├── _common.py      # Shared helpers (arg groups, pipeline detection, loading, remote, I/O)
├── image.py        # text-to-image, image-to-image, inpaint
├── video.py        # text-to-video, image-to-video
├── audio.py        # text-to-audio
├── modular.py      # generic ModularPipeline runner with free-form inputs
└── tasks.py        # `tasks` — list every registered agentic command
```

## Discovering tasks

```bash
diffusers-cli tasks            # human-readable
diffusers-cli tasks --json     # for agents
```

## Pipeline detection (DiffusionPipeline vs ModularPipeline)

Every inference command auto-detects whether the `--model` is a regular
`DiffusionPipeline` repo (`model_index.json`) or a custom
`ModularPipeline` repo (`modular_model_index.json`) via a single Hub
listing — no weights are downloaded. If you point a task-shaped command at
a modular repo, it exits with a hint to use `diffusers-cli modular`
instead. The reverse is also true: `modular` rejects a regular repo and
points back at the task-shaped command.

## Pushing outputs to a bucket

Every inference command (and `modular`) accepts `--push-to <bucket_id>`
to upload the generated files to a Hugging Face **bucket** after they're
saved locally. The bucket is created if it doesn't exist and files land
under a prefix named after the task (e.g. `text-to-image/<filename>`).

```bash
diffusers-cli text-to-image \
  --model stabilityai/stable-diffusion-xl-base-1.0 \
  --prompt "a watercolor of a fox" \
  --num-images 4 \
  --push-to your-username/cli-generations
```

The upload is a single `batch_bucket_files` round-trip regardless of how
many files were generated. The JSON payload reports `hf://buckets/...`
URIs so an agent can pipe them into a follow-up tool.

## Running remotely (HF Jobs) and fetching outputs back

Every inference command supports `--remote`, which submits the same call
to Hugging Face Jobs via `huggingface_hub.run_uv_job`, then by default
**waits for the job to finish and downloads the outputs back to your
local machine**.

The flow:

1. If `--push-to` isn't set, default it to `<your-user>/jobs-artifacts`
   (the canonical jobs bucket — `https://huggingface.co/buckets/<you>/jobs-artifacts`).
2. Generate a random `run_id` and pass it via `DIFFUSERS_CLI_RUN_ID` env
   so the container writes its files under `<run_id>/` inside the bucket.
3. Submit the job (your `HF_TOKEN` is forwarded as a secret).
4. Poll `inspect_job` every `--poll-interval` seconds until the stage is
   `COMPLETED` / `CANCELED` / `ERROR` / `DELETED`.
5. List `<run_id>/` in the bucket and `download_bucket_files` everything
   into the local `--output` directory (default `./outputs/`).

Pass `--no-wait` to fire-and-forget — the command prints the job id and
returns immediately; you can fetch later via `huggingface-cli buckets`.

| Option | Description |
|--------|-------------|
| `--remote` | Run on HF Jobs instead of locally |
| `--flavor` | Hardware flavor (default `a10g-small`) |
| `--timeout` | Job timeout (e.g. `30m`, `2h`) |
| `--dependencies` | Extra pip deps. Repeat for multiple |
| `--namespace` | HF namespace (defaults to the current user) |
| `--no-wait` | Skip polling/download — submit and exit |
| `--poll-interval` | Seconds between job-status polls (default 5) |

```bash
# Submit text-to-image to HF Jobs on an A100, wait, download to ./outputs/
diffusers-cli text-to-image \
  --model stabilityai/stable-diffusion-xl-base-1.0 \
  --prompt "a watercolor of a fox in autumn leaves" \
  --num-images 4 \
  --remote --flavor a100-large --timeout 30m
```

```bash
# Same call, fire-and-forget
diffusers-cli text-to-image ... --remote --no-wait
```

## Common options

Every inference command supports:

| Option | Description |
|--------|-------------|
| `--model` / `-m` | Model id on the Hub or local path |
| `--device` | `cpu`, `cuda`, `cuda:0`, `mps` (defaults to best available) |
| `--dtype` | `auto`, `float16`, `bfloat16`, `float32` |
| `--variant` | Optional weight variant (e.g. `fp16`) |
| `--revision` | Model revision (branch, tag, or SHA) |
| `--token` | Hugging Face token for gated/private models |
| `--trust-remote-code` | Allow custom code from the Hub |
| `--output` / `-o` | Output file or directory |
| `--json` | Machine-readable JSON summary on stdout |
| `--seed` | Random seed for reproducibility |
| `--pipeline-kwargs` | JSON object of extra kwargs forwarded to the pipeline call |

## Commands

### Image

1. Generate an image from a text prompt
   ```bash
   diffusers-cli text-to-image \
     --model stabilityai/stable-diffusion-xl-base-1.0 \
     --prompt "a watercolor of a fox in autumn leaves" \
     --output fox.png
   ```

2. Generate with explicit sampling controls
   ```bash
   diffusers-cli text-to-image \
     --model stabilityai/stable-diffusion-xl-base-1.0 \
     --prompt "studio portrait of a cyberpunk hacker" \
     --negative-prompt "blurry, low quality" \
     --num-inference-steps 30 \
     --guidance-scale 7.5 \
     --height 1024 --width 1024 \
     --seed 42
   ```

3. Generate multiple variants at once
   ```bash
   diffusers-cli text-to-image \
     --model black-forest-labs/FLUX.1-schnell \
     --prompt "a still life with citrus and ceramics" \
     --num-images 4 \
     --output ./outputs/still-life/
   ```

4. Transform an existing image with a prompt (image-to-image)
   ```bash
   diffusers-cli image-to-image \
     --model stabilityai/stable-diffusion-xl-refiner-1.0 \
     --image input.jpg \
     --prompt "make it look like an oil painting" \
     --strength 0.6 \
     --output painted.png
   ```

5. Inpaint a masked region of an image
   ```bash
   diffusers-cli inpaint \
     --model stabilityai/stable-diffusion-2-inpainting \
     --image photo.png \
     --mask mask.png \
     --prompt "a golden retriever sitting on the bench" \
     --output filled.png
   ```

6. Emit JSON for downstream tooling
   ```bash
   diffusers-cli text-to-image \
     --model stabilityai/sdxl-turbo \
     --prompt "neon city at night" \
     --json
   ```

### Video

7. Generate a short clip from a text prompt
   ```bash
   diffusers-cli text-to-video \
     --model THUDM/CogVideoX-2b \
     --prompt "a panda surfing on a wave at sunset" \
     --num-frames 49 \
     --fps 8 \
     --output panda.mp4
   ```

8. Animate a single still image
   ```bash
   diffusers-cli image-to-video \
     --model stabilityai/stable-video-diffusion-img2vid-xt \
     --image still.png \
     --prompt "subtle camera dolly forward" \
     --num-frames 25 \
     --output animated.mp4
   ```

### Audio

9. Generate music or a sound effect from a text prompt
   ```bash
   diffusers-cli text-to-audio \
     --model cvssp/audioldm2 \
     --prompt "a calm piano melody in a quiet room" \
     --audio-length-in-s 10 \
     --output music.wav
   ```

### Modular pipelines

Modular pipelines have an open-ended input surface defined by the block
graph, so the CLI doesn't try to predict it — pass inputs verbatim.

14. Run a modular pipeline with free-form inputs
    ```bash
    diffusers-cli modular \
      --model your-username/my-modular-pipeline \
      --inputs prompt="a calm landscape" \
      --inputs num_inference_steps=25 \
      --inputs-json '{"guidance_scale": 4.5}' \
      --output-key image \
      --output out.png
    ```

The output type is auto-detected — a PIL image (or list of PIL images)
becomes PNG(s), a sequence of frames becomes an MP4, a numpy audio array
becomes a WAV, and anything else is JSON-serialized.

### Roadmap

Open an issue if you'd like to help land one:

- **Video**: `video-to-video`
- **Conditioning**: ControlNet, T2I-Adapter, instruction editing (Flux-Kontext, InstructPix2Pix)
- **Quantization / export**: `convert` (fp16/safetensors/GGUF), `quantize` (bitsandbytes, torchao)
