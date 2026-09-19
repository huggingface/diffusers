# MAGI-1 base inference

This example loads a converted Diffusers checkpoint and uses the sampling parameters from an official MAGI
base config. It does not import the official implementation. The default prompt, `Good Boy`, is the prompt in
the official 4.5B `example/4.5B/run.sh`.

## Environment

The tested base environment is Python 3.10, PyTorch 2.5.1+cu121, torchvision 0.20.1+cu121, and Flash Attention
2.7.4.post1 on CUDA. Flash Attention must be built for the installed PyTorch/CUDA combination.
The requirements file pins the additional Python dependencies. It does not install or replace PyTorch or
Flash Attention. To preserve an existing compatible environment, create a separate overlay from its Python:

```bash
/path/to/diffusion/bin/python -m venv --system-site-packages /path/to/magi-env
/path/to/magi-env/bin/python -m pip install -r examples/magi/requirements.txt
```

Run commands below from the Diffusers repository root with `PYTHONPATH=src`. This ensures the local MAGI
implementation is used without changing the installed Diffusers package in the base environment.
The overlay still depends on its base environment; it is not a standalone container or a complete system lock.

For repository checks, install `requirements-dev.txt` instead and activate the overlay so subprocesses find
the pinned Ruff and documentation builder:

```bash
/path/to/magi-env/bin/python -m pip install -r examples/magi/requirements-dev.txt
source /path/to/magi-env/bin/activate
make quality
```

## Official text-to-video example

First convert the original checkpoint with `scripts/convert_magi_to_diffusers.py pipeline` as described in the
MAGI modular pipeline documentation. Then run:

```bash
PYTHONPATH=src /path/to/magi-env/bin/python examples/magi/inference_magi.py \
    --model /path/to/MAGI-1-diffusers \
    --config /path/to/MAGI-1/example/4.5B/4.5B_base_config.json \
    --output /path/to/new-output-directory
```

The official 4.5B base config selects 720 × 720, 96 frames, 64 steps, seed 1234, and 24 fps. The example maps
the config's guidance thresholds/scales and clean-prefix attention settings to the Diffusers components.
Official checkpoint paths and distributed-engine settings in the config are not used: `--model` selects the
converted weights, and this example runs on one GPU with component CPU offload.

Use `--device cuda:1` to select another GPU. `--height`, `--width`, `--num-frames`, `--seed`, and `--prompt`
explicitly override the official sampling inputs. Overrides are recorded in the output metadata.
For example, `--height 512 --width 512 --num-frames 192 --seed 1235` tests a longer clip.

The output directory must be new or empty. It receives `settings.json`, `output_t2v.mp4`, and `metrics.json`.
Use `--save-latents` to also retain the final latent tensor. Generation time includes text encoding, denoising,
and VAE decoding, but excludes loading and MP4 encoding. MP4 frame count and finite/range checks run before
the final metrics are written. A successful run is not, by itself, a visual-quality or official numerical-parity
claim.

## Precision and scope

T5 and sampling state remain FP32; the Transformer and VAE load in BF16 with the model's FP32 exceptions.
The Transformer uses `flash_varlen`, the VAE uses `flash`, and TF32 is disabled. VAE temporal tiles use
`fps // 2` input frames (12 at 24 fps), matching the official pipeline entry point rather than the VAE helper's
16-frame default. The temporal tile size is recorded in the output metadata.
Changing backend, precision, or generator device can change the output.

This entry point supports non-quantized base T2V, I2V, and V2V. See [README-prefix.md](README-prefix.md) for
image/video prefix inputs and output-length conventions. Distilled sampling, FP8, and distributed execution
are not supported by this example.
