# EdenLive inference

Generate video from a reference image, proxy video, and prompt. See the [pipeline guide](../../docs/source/en/api/pipelines/edenlive.md)
for the Python API and model defaults.

## Setup

Install a CUDA-compatible [PyTorch](https://pytorch.org/get-started/locally/) and torchvision, then run from the
Diffusers repository root:

```bash
pip install -e . "transformers>=4.56.0" accelerate sentencepiece ftfy av imageio imageio-ffmpeg
```

The [requirements file](requirements.txt) also lists the example dependencies.

## Single GPU

```bash
python examples/edenlive/inference_edenlive.py \
  --model VAST-AI/EdenLive-5B-Causal-Diffusers --causal \
  --video proxy.mp4 --image reference.png \
  --prompt "A character explores a snowy landscape." \
  --num-frames 49
```

Use `--causal` with a causal repository, or omit it for a bidirectional repository. `--model` also accepts a local
Diffusers directory. The script resamples the proxy video to the model's default frame rate or `--fps`.
`--num-frames` must be $4k+1$ and fit within the resampled video; omit it to use the largest valid prefix.

The default output is `edenlive.mp4`, with timing and input details in `edenlive.json`. Set `--output` to change it.
Use `--lighttae` for faster decoding, `--vae-tiling` to reduce VAE memory use, or `--cpu-offload` to move models
to the GPU only when needed. Run
`python examples/edenlive/inference_edenlive.py --help` for all options.

## Model repositories

| Size | Bidirectional | Causal |
|---|---|---|
| 5B | [EdenLive-5B-Diffusers](https://huggingface.co/VAST-AI/EdenLive-5B-Diffusers) | [EdenLive-5B-Causal-Diffusers](https://huggingface.co/VAST-AI/EdenLive-5B-Causal-Diffusers) |
| 14B | [EdenLive-14B-Diffusers](https://huggingface.co/VAST-AI/EdenLive-14B-Diffusers) | [EdenLive-14B-Causal-Diffusers](https://huggingface.co/VAST-AI/EdenLive-14B-Causal-Diffusers) |

Each repository contains one complete pipeline. Select the matching repository and pipeline variant.

## Changing prompts

With `--causal`, add `--chunk-prompts prompts.json`. For 49 frames, supply five prompts:

```json
[
  "A character walks through snow.",
  "A character walks through snow.",
  "The character approaches a campfire.",
  "The character warms their hands beside the campfire.",
  "The character rests beside the campfire."
]
```

The list replaces `--prompt`, with one prompt per chunk. The negative prompt remains fixed.

## Convert original checkpoints

To convert a local original release instead of loading the published Diffusers weights:

```bash
python scripts/convert_edenlive_to_diffusers.py \
  --source /path/to/EdenLive --size 5b --variant bidirectional \
  --output /path/to/EdenLive-5B-Diffusers
```

Use `--size 14b` for 14B or `--variant causal` for causal generation, with a separate output directory for each
combination. Each package contains one sampling variant and the optional LightTAE decoder.
