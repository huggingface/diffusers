# Cosmos3 Inference

This example shows how to run inference with the Cosmos3 Omni pipeline — NVIDIA's Mixture-of-Transformer (MoT) world foundation model — using the diffusers library.

## Setup

Install the example's dependencies:

```bash
pip install -r examples/cosmos3/requirements.txt
```

## Running inference

The script downloads the pipeline from the [HuggingFace Hub](https://huggingface.co/nvidia/Cosmos3-Nano/tree/main/):

```bash
CUDA_VISIBLE_DEVICES=0 python examples/cosmos3/inference_cosmos3.py \
    --input examples/cosmos3/inputs/omni/i2v.json
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--input` | `inputs/omni/i2v.json` | JSON file with `prompt` and optional `vision_path` |
| `--output` | `.` | Directory to write output files |
| `--height` | `720` | Output height in pixels |
| `--width` | `1280` | Output width in pixels |
| `--num-frames` | from JSON, else `189` | Number of frames (`1` = image) |

## Example inputs

The `inputs/` directory contains ready-to-use examples:

| File | Mode |
|---|---|
| `inputs/omni/t2i.json` | Text-to-image (`"num_frames": 1`) |
| `inputs/omni/t2v.json` | Text-to-video |
| `inputs/omni/i2v.json` | Image-to-video |

Each JSON requires a `"prompt"` field and optionally a `"vision_path"` (URL or local path) for image conditioning.
