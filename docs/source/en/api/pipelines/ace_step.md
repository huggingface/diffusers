<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# ACE-Step 1.5

ACE-Step 1.5 was proposed in [ACE-Step: A Step Towards Music Generation Foundation Model](https://arxiv.org/abs/2602.00744) by the ACE-Step Team. It is a highly efficient open-source music foundation model that generates commercial-grade music with lyrics from text prompts.

ACE-Step 1.5 generates variable-length stereo music at 48kHz (from 10 seconds to 10 minutes) from text prompts and optional lyrics. It comprises three components: an Oobleck autoencoder (VAE) that compresses waveforms into 25Hz latent representations, a Qwen3-based text encoder for text and lyric conditioning, and a Diffusion Transformer (DiT) model that operates in the latent space of the autoencoder using flow matching.

The model supports 50+ languages for lyrics including English, Chinese, Japanese, Korean, French, German, Spanish, Italian, Portuguese, Russian, and more. It runs locally with less than 4GB of VRAM and generates a full song in under 2 seconds on an A100.

This pipeline was contributed by [ACE-Step Team](https://github.com/ACE-Step). The original codebase can be found at [ACE-Step/ACE-Step-1.5](https://github.com/ACE-Step/ACE-Step-1.5).

## Tips

When constructing a prompt, keep in mind:

* Descriptive prompt inputs work best; use adjectives to describe the music style, instruments, mood, and tempo.
* The prompt should describe the overall musical characteristics (e.g., "upbeat pop song with electric guitar and drums").
* Lyrics should be structured with tags like `[verse]`, `[chorus]`, `[bridge]`, etc.

During inference:

* The turbo model variant is designed for 8 inference steps with `shift=3.0`.
* The `audio_duration` parameter controls the length of the generated music in seconds.
* The `vocal_language` parameter should match the language of the lyrics.

```python
import torch
import soundfile as sf
from diffusers import AceStepPipeline

pipe = AceStepPipeline.from_pretrained("ACE-Step/ACE-Step-v1-5-turbo", torch_dtype=torch.bfloat16)
pipe = pipe.to("cuda")

audio = pipe(
    prompt="A beautiful piano piece with soft melodies and gentle rhythm",
    lyrics="[verse]\nSoft notes in the morning light\nDancing through the air so bright\n[chorus]\nMusic fills the air tonight\nEvery note feels just right",
    audio_duration=30.0,
    num_inference_steps=8,
).audios

sf.write("output.wav", audio[0].T.cpu().float().numpy(), 48000)
```

## AceStepPipeline
[[autodoc]] AceStepPipeline
	- all
	- __call__
