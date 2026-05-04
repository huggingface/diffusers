<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# ACE-Step 1.5

ACE-Step 1.5 was introduced in [ACE-Step 1.5: Pushing the Boundaries of Open-Source Music Generation](https://arxiv.org/abs/2602.00744) by the ACE-Step Team (ACE Studio and StepFun). It is an open-source music foundation model that generates commercial-grade stereo music with lyrics from text prompts.

ACE-Step 1.5 generates variable-length stereo audio at 48 kHz (10 seconds to 10 minutes) from text prompts and optional lyrics. The full system pairs a Language Model planner with a Diffusion Transformer (DiT) synthesizer; this pipeline wraps the DiT half of that stack, and consists of three components: an [`AutoencoderOobleck`] VAE that compresses waveforms into 25 Hz stereo latents, a Qwen3-based text encoder for prompt and lyric conditioning, and an [`AceStepTransformer1DModel`] DiT that operates in the VAE latent space using flow matching.

The model supports 50+ languages for lyrics — including English, Chinese, Japanese, Korean, French, German, Spanish, Italian, Portuguese, and Russian — and runs on consumer GPUs (under 4 GB of VRAM when offloaded).

This pipeline was contributed by the [ACE-Step Team](https://github.com/ace-step). The original codebase can be found at [ace-step/ACE-Step-1.5](https://github.com/ace-step/ACE-Step-1.5).

## Variants

ACE-Step 1.5 ships three DiT checkpoints that share the same transformer architecture but differ in guidance behavior; the pipeline auto-detects turbo checkpoints from the loaded transformer config and ignores CFG guidance for those guidance-distilled weights.

| Variant | CFG | Default steps | Default `guidance_scale` | Default `shift` | HF repo |
|---------|:---:|:-------------:|:------------------------:|:---------------:|---------|
| `turbo` (guidance-distilled) | off | 8 | ignored | 3.0 | [`ACE-Step/Ace-Step1.5`](https://huggingface.co/ACE-Step/Ace-Step1.5) |
| `base` | on | 8 | 7.0 | 3.0 | [`ACE-Step/acestep-v15-base`](https://huggingface.co/ACE-Step/acestep-v15-base) |
| `sft` | on | 8 | 7.0 | 3.0 | [`ACE-Step/acestep-v15-sft`](https://huggingface.co/ACE-Step/acestep-v15-sft) |

Base and SFT use the learned `null_condition_emb` for classifier-free guidance (APG, not vanilla CFG). Users commonly override `num_inference_steps` to 30–60 on base/sft for higher quality.

## Tips

When constructing a prompt, keep in mind:

* Descriptive prompt inputs work best; use adjectives to describe the music style, instruments, mood, and tempo.
* The prompt should describe the overall musical characteristics (e.g., "upbeat pop song with electric guitar and drums").
* Lyrics should be structured with tags like `[verse]`, `[chorus]`, `[bridge]`, etc.

During inference:

* `num_inference_steps`, `guidance_scale`, and `shift` default to the values shown above. For turbo checkpoints, `guidance_scale > 1.0` is ignored with a warning because guidance is distilled into the weights.
* The `audio_duration` parameter controls the length of the generated music in seconds.
* The `vocal_language` parameter should match the language of the lyrics.
* `pipe.sample_rate` and `pipe.latents_per_second` are sourced from the VAE config (48000 Hz and 25 fps for the released checkpoints).
* For audio-to-audio tasks, pass `src_audio` and `reference_audio` as preprocessed stereo tensors at `pipe.sample_rate`.
* `flash` and `flash_hub` use FlashAttention's native sliding-window support for ACE-Step's self-attention and expect unpadded text batches. If a batched prompt contains padding, use `flash_varlen` or `flash_varlen_hub` instead. Single-prompt inference with `padding="longest"` is normally unpadded.

```python
import torch
import soundfile as sf
from diffusers import AceStepPipeline

pipe = AceStepPipeline.from_pretrained("ACE-Step/Ace-Step1.5", torch_dtype=torch.bfloat16)
pipe = pipe.to("cuda")

audio = pipe(
    prompt="A beautiful piano piece with soft melodies and gentle rhythm",
    lyrics="[verse]\nSoft notes in the morning light\nDancing through the air so bright\n[chorus]\nMusic fills the air tonight\nEvery note feels just right",
    audio_duration=30.0,
).audios

sf.write("output.wav", audio[0].T.cpu().float().numpy(), pipe.sample_rate)
```

## AceStepPipeline
[[autodoc]] AceStepPipeline
	- all
	- __call__
