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

This pipeline supports loading the original flat LongCat checkpoint layout from either a local directory or a Hugging Face Hub repository containing:

- `config.json`
- `model.safetensors`

The loader builds the text encoder, transformer, and VAE from `config.json`, restores component weights from `model.safetensors`, and ties the shared UMT5 embedding when needed.

This pipeline was adapted from the LongCat-AudioDiT reference implementation: https://github.com/meituan-longcat/LongCat-AudioDiT

## Usage

```py
import soundfile as sf
import torch
from diffusers import LongCatAudioDiTPipeline

pipeline = LongCatAudioDiTPipeline.from_pretrained(
    "meituan-longcat/LongCat-AudioDiT-1B",
    torch_dtype=torch.float16,
)
pipeline = pipeline.to("cuda")

audio = pipeline(
    prompt="A calm ocean wave ambience with soft wind in the background.",
    audio_end_in_s=5.0,
    num_inference_steps=16,
    guidance_scale=4.0,
    output_type="pt",
).audios

output = audio[0, 0].float().cpu().numpy()
sf.write("longcat.wav", output, pipeline.sample_rate)
```

## Tips

- `audio_end_in_s` is the most direct way to control output duration.
- `output_type="pt"` returns a PyTorch tensor shaped `(batch, channels, samples)`.

## LongCatAudioDiTPipeline

[[autodoc]] LongCatAudioDiTPipeline
	- all
	- __call__
	- from_pretrained
