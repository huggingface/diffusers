<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# MiniMax Music 3

[MiniMax Music 3](https://huggingface.co/MiniMaxAI/MiniMax-Music3) is a music generation model that produces complete
songs up to five minutes long from lyrics and a music description, with expressive vocals and long-range structure.

The model is a hybrid of an autoregressive and a diffusion stage: an 8B Qwen3-based global language model predicts one
semantic audio token per frame while a small depth decoder fills in seven residual RVQ codebooks, and their fused
hidden states condition a 2.4B flow-matching transformer that produces Flow-VAE latents in overlapping chunks. A
DAC-style decoder turns the latents into 44.1 kHz stereo audio.

## Usage

MiniMax Music 3 is available as a modular pipeline.

```py
import soundfile as sf
import torch
from diffusers import ModularPipeline

pipe = ModularPipeline.from_pretrained("MiniMaxAI/MiniMax-Music3")
pipe.load_components(dtype=torch.bfloat16)
pipe.to("cuda")  # or "mps", "xpu", "cpu"

lyrics = """[verse]
Morning light filtering through the pine
Every quiet street is yours and mine
[chorus]
Softly the world begins to breathe"""

prompt = (
    "Genre: acoustic pop. BPM: 96. Key: C major. Warm and intimate, building gently into the chorus. "
    "Vocals: soft female lead, close and breathy, light stacked harmonies in the chorus. "
    "Arrangement: fingerpicked guitar and soft piano; brushed drums and upright bass enter in the chorus."
)

audio = pipe(
    prompt=prompt,
    lyrics=lyrics,
    audio_duration=60.0,
    generator=torch.Generator("cuda").manual_seed(7),
    output="audios",
)[0]

sf.write("minimax_music3.wav", audio.T, pipe.sampling_rate)
```

## Reduce memory usage

Refer to the [Reduce memory usage](../../optimization/memory) guide for more details about the various memory saving
techniques.

The full pipeline needs ~23 GB of VRAM in bfloat16. With automatic CPU offloading a generation runs in ~22 GB of free
VRAM, and additionally group-offloading the language model fits in 8 GB.

```py
import torch
from diffusers import ComponentsManager, ModularPipeline
from diffusers.hooks.group_offloading import apply_group_offloading

manager = ComponentsManager()
manager.enable_auto_cpu_offload(device="cuda")  # or "mps", "xpu", "cpu"
pipe = ModularPipeline.from_pretrained("MiniMaxAI/MiniMax-Music3", components_manager=manager)
pipe.load_components(dtype=torch.bfloat16)

# Only needed below ~22 GB of free VRAM — slower, but fits in 8 GB.
apply_group_offloading(
    pipe.language_model, onload_device=torch.device("cuda"), offload_type="leaf_level", use_stream=True
)
```

## Tips

- Structure tags such as `[intro]`, `[verse]`, `[pre-chorus]`, `[chorus]`, `[bridge]`, `[instrumental]`, `[solo]`, and
  `[outro]` must each be on their own line in `lyrics`. Text on the same line as a leading tag is dropped by the
  model's input contract.
- The music description controls the vocals: describe the vocal gender and timbre explicitly (e.g. "warm female
  vocal") or the model may drift instrumental. For fine-grained control, structure the description into global
  metadata (genre, BPM, key, emotional progression), vocal details, and arrangement.
- `audio_duration` is an upper bound — the language model may end the song earlier with a stop token. The
  autoregressive stage generates 25 frames per second of audio and dominates the runtime.
- The classifier-free guidance scale of the flow-matching stage is a guider setting (the reference inference value is
  1.7): swap it with `pipe.update_components(guider=ClassifierFreeGuidance(guidance_scale=...))`.
- The pipeline returns the vocoder's native 44.1 kHz stereo output. The reference server additionally resamples to 32
  kHz; apply your own resampling if you need that exact rate.

## MiniMaxMusic3ModularPipeline

[[autodoc]] MiniMaxMusic3ModularPipeline

## MiniMaxMusic3Blocks

[[autodoc]] MiniMaxMusic3Blocks

## MiniMaxMusic3ConditionEncoder

[[autodoc]] MiniMaxMusic3ConditionEncoder

## MiniMaxMusic3RVQDepthDecoder

[[autodoc]] MiniMaxMusic3RVQDepthDecoder

## MiniMaxMusic3Vocoder

[[autodoc]] MiniMaxMusic3Vocoder
