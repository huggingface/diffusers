<!--Copyright 2025 Stability AI and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Stable Audio 3

Stable Audio 3 (SA3) is a text-to-audio model from [Stability AI](https://stability.ai/) that generates high-quality
stereo audio at 44.1 kHz. It uses a rectified-flow DiT conditioned on two signals:

* **Text** — encoded by a frozen T5Gemma encoder and injected via cross-attention.
* **Duration** — a float (seconds) embedded by [`StableAudio3DurationEmbedder`] and used as a global conditioning
  vector for adaptive layer normalisation.

Audio is decoded by the SAME (Semantically-Aligned Music Encoder) autoencoder, [`AutoencoderSAME`].

Both checkpoints use [`FlowMatchEulerDiscreteScheduler`] with a log-SNR-uniform sigma schedule, differing only in
`stochastic_sampling` and the default step count:

| Checkpoint | `diffusion_objective` | `stochastic_sampling` | `num_inference_steps` |
|---|---|---|---|
| `stable-audio-3-medium-base` | `rectified_flow` | `False` (deterministic Euler) | **100** (not distilled) |
| `stable-audio-3-medium` (distilled) | `rf_denoiser` | `True` (ping-pong re-noise) | **8** (distilled for 8 steps) |

The correct scheduler config is baked into each converted checkpoint, so `num_inference_steps` defaults to the right
value when you leave it unset. Only pass it to override.

Original codebase: [Stability-AI/stable-audio-3](https://github.com/Stability-AI/stable-audio-3).

## Converting original checkpoints

The Stability AI checkpoints are not published in diffusers format, so convert them locally. The script downloads the
checkpoint's `model_config.json` and selects the scheduler from its `diffusion_objective`:

```bash
python scripts/convert_stable_audio_3_to_diffusers.py \
  --checkpoint_path stabilityai/stable-audio-3-medium-base \
  --text_encoder_repo google/t5gemma-b-b-ul2 \
  --output_dir /tmp/sa3-diffusers-euler \
  --dtype float32
```

> [!TIP]
> `stable-audio-3-medium-base` is a **gated** repo. Run `hf auth login` with an account that has access before
> converting, otherwise the download fails with a 401.

## Usage example

Load the converted checkpoint from its local output directory (install
[`soundfile`](https://pypi.org/project/soundfile/) with `pip install soundfile`):

```py
import torch
import soundfile as sf
from diffusers import StableAudio3Pipeline

pipe = StableAudio3Pipeline.from_pretrained("/tmp/sa3-diffusers-euler", torch_dtype=torch.float32)
pipe = pipe.to("cuda")  # or "mps", "xpu", "cpu"

generator = torch.Generator("cuda").manual_seed(0)
audio = pipe(
    "A gentle piano melody with soft strings in a concert hall",
    duration=10.0,  # seconds; latent length is computed automatically
    generator=generator,
).audios

sf.write("sa3_output.wav", audio[0].T.cpu().float().numpy(), samplerate=44100)
```

The pipeline is also registered with [`AutoPipelineForText2Audio`], which resolves the checkpoint to
`StableAudio3Pipeline` automatically:

```py
from diffusers import AutoPipelineForText2Audio

pipe = AutoPipelineForText2Audio.from_pretrained("/tmp/sa3-diffusers-euler", torch_dtype=torch.float32)
```

> [!NOTE]
> The examples use a local path because `stabilityai/stable-audio-3-medium` and `stable-audio-3-medium-base` are not
> yet published in diffusers format (loading by repo id returns a 404). Once published, the repo id works in place of
> the local path.

## Tips

* Use `torch.float32` on CPU or MPS (Apple Silicon) — `torch.float16` on MPS produces noise.
* The distilled model (`stable-audio-3-medium`) is **adversarially distilled** — guidance is baked into the weights.
  Leave `guidance_scale=1.0` (the default) and don't pass a `negative_prompt` for that checkpoint; both only do
  something useful for the non-distilled `stable-audio-3-medium-base` checkpoint.
* `silence_padding_duration` (default `0.0`) adds silent headroom at the end of the latent sequence. Leave it at `0.0`
  unless the model is trained to mask that padding — otherwise the extra frames drain output energy and the result
  gets quiet.
* Set `num_waveforms_per_prompt > 1` to generate multiple clips per prompt.

## StableAudio3Pipeline

[[autodoc]] StableAudio3Pipeline
	- all
	- __call__

## StableAudio3InpaintPipeline

[[autodoc]] StableAudio3InpaintPipeline
	- all
	- __call__

## StableAudio3AudioToAudioPipeline

Generates a variation of a reference audio clip: the whole reference is noised to `init_noise_level` and denoised
from there, unlike [`StableAudio3InpaintPipeline`]'s per-frame local-additive conditioning which preserves specific
frames exactly.

[[autodoc]] StableAudio3AudioToAudioPipeline
	- all
	- __call__

## StableAudio3DurationEmbedder

[[autodoc]] StableAudio3DurationEmbedder
