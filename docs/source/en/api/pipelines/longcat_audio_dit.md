<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# LongCat-AudioDiT

LongCat-AudioDiT is a text-to-audio diffusion model from Meituan LongCat. The diffusers integration exposes a standard [`DiffusionPipeline`] interface for text-conditioned audio generation.

This pipeline was adapted from the LongCat-AudioDiT reference implementation: https://github.com/meituan-longcat/LongCat-AudioDiT

This pipeline supports loading from a local directory or Hugging Face Hub repository in diffusers format (containing `text_encoder/`, `transformer/`, `vae/`, `tokenizer/`, and `scheduler/` subfolders).

## Usage

```py
import soundfile as sf
import torch
from diffusers import LongCatAudioDiTPipeline

pipeline = LongCatAudioDiTPipeline.from_pretrained(
    "ruixiangma/LongCat-AudioDiT-1B-Diffusers",
    torch_dtype=torch.float16,
)
pipeline = pipeline.to("cuda")

prompt = "A calm ocean wave ambience with soft wind in the background."
audio = pipeline(
    prompt,
    audio_duration_s=5.0,
    num_inference_steps=16,
    guidance_scale=4.0,
    generator=torch.Generator("cuda").manual_seed(42),
).audios[0, 0]

sf.write("longcat.wav", audio, pipeline.sample_rate)
```

## Tips

- `audio_duration_s` is the most direct way to control output duration.
- Use `generator=torch.Generator("cuda").manual_seed(42)` to make generation reproducible.
- Output shape is `(batch, channels, samples)` - use `.audios[0, 0]` to get a single audio sample.
- The pipeline outputs mono audio (1 channel). If you need stereo, you can duplicate the channel: `audio.unsqueeze(0).repeat(1, 2, 1)`.

## LongCatAudioDiTPipeline

[[autodoc]] LongCatAudioDiTPipeline
	- all
	- __call__
	- from_pretrained
