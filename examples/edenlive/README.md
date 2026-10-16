# EdenLive inference

Render game from a reference image, prompt, and proxy video with `EdenLivePipeline`.
See the [pipeline guide](../../docs/source/en/api/pipelines/edenlive.md) for the Python API.

## Setup

Install a CUDA-compatible [PyTorch](https://pytorch.org/get-started/locally/) build and matching torchvision.
From the Diffusers repository root, install the pipeline dependencies:

```bash
pip install -e . "transformers>=4.56.0" accelerate sentencepiece ftfy
```

For the CLI examples, add video reading and writing:

```bash
pip install av imageio imageio-ffmpeg
```

The [example requirements](requirements.txt) collect both groups, plus PyTorch and torchvision.

## Single GPU

```bash
python examples/edenlive/inference_edenlive.py \
  --model VAST-AI/EdenLive --profile 5b-causal \
  --video proxy.mp4 --image reference.png \
  --prompt "A character explores a snowy landscape." \
  --num-frames 49 --output edenlive.mp4
```

`--model` accepts a Hub repository, local release, or saved Diffusers pipeline. Select its profile with `--profile`.
Use `--local-files-only` for local or cached weights.

| Profile | Default height x width | Default fps | Default steps | Guidance scale |
|---|---|---|---|---|
| `5b-bidir` | 704 x 1280 | 24 | 48 | 3 |
| `14b-bidir` | 480 x 832 | 16 | 30 | 3 |
| `5b-causal` | 704 x 1280 | 24 | 4 per chunk | 1 |
| `14b-causal` | 480 x 832 | 16 | 4 per chunk | 1 |

The CLI resamples to `--fps`, center-crops, and resizes the input. Set `--num-frames` to a positive $4k+1$ within
the resampled clip, or omit it to use the largest valid prefix. Custom height and width use multiples of 32 for 5B
and 16 for 14B.

Causal profiles use four fixed steps per chunk. Bidirectional profiles support `--num-inference-steps`.
Set `--guidance-scale` above one to use `--negative-prompt`.

## Multiple GPUs

Use two or four GPUs with Ulysses context parallelism and PyTorch FSDP2 (`torch.distributed.fsdp.fully_shard`):

```bash
torchrun --standalone --nproc_per_node=2 examples/edenlive/inference_edenlive.py \
  --model VAST-AI/EdenLive --profile 14b-causal \
  --video proxy.mp4 --image reference.png \
  --prompt "A character explores a snowy landscape." \
  --num-frames 81 --output edenlive.mp4
```

Set `--nproc_per_node=4` for four GPUs. Add `--no-fsdp` to use context parallelism with replicated transformer weights.

## Temporal prompts

For `5b-bidir`, pass `--text-timeline timeline.json`. This example describes 49 frames at 24 fps:

```json
{
  "fps": 24,
  "num_frames": 49,
  "segments": [
    {"start_sec": 0.0, "end_sec": 1.0, "text": "A character walks through snow."},
    {"start_sec": 1.0, "end_sec": 2.0, "text": "The character approaches a warm campfire."}
  ]
}
```

Match the JSON `fps` and `num_frames` to the CLI values. Use contiguous segments covering zero through the last
frame timestamp, with at least one latent frame per segment.

For either causal profile, pass `--chunk-prompts prompts.json` with one prompt per chunk. A 49-frame video uses
five chunks, including a shorter final chunk:

```json
[
  "A character walks through snow.",
  "A character walks through snow.",
  "The character approaches a campfire.",
  "The character warms their hands beside the campfire.",
  "The character rests beside the campfire."
]
```

Include `--prompt` in the command; the timeline or chunk prompts supply the text used during generation.

## Streaming from proxy images

`inference_edenlive_streaming.py` supports both causal profiles. Pass ordered RGB images with `--proxy-frames`,
already sampled at latent cadence: three images per chunk.

```bash
python examples/edenlive/inference_edenlive_streaming.py \
  --model /path/to/EdenLive --profile 5b-causal \
  --image reference.png --prompt "A character walks through snow" \
  --proxy-frames proxy/000.png proxy/004.png proxy/008.png proxy/012.png proxy/016.png proxy/020.png \
  --chunk-prompts stream_prompts.json --output stream.mp4 --fps 24 --local-files-only
```

The optional `stream_prompts.json` contains one string per chunk, for example
`["A character walks through snow", "A character walks through fire"]`. Omit it to use `--prompt` throughout.

The script writes video incrementally: 9 frames for the first chunk, then 12 per chunk. Set `--fps` to the desired
playback rate (24 by default). For application integration, see the
[streaming API](../../docs/source/en/api/pipelines/edenlive.md#streaming-causal-video).

## Offline options and reports

LightTAE is the default decoder for original releases. Use `--full-vae` for Wan VAE decoding and `--vae-tiling` for
tiling on the Wan VAE. Set the noise seed with `--seed` and the attention backend with `--attention-backend`.
Run `python examples/edenlive/inference_edenlive.py --help` for all options.

The offline script writes a JSON report beside the video with arguments, sampled frame indices, timing, and peak GPU
memory. Timing covers the pipeline call, including conditioning and decoding.
