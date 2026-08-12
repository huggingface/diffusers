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

```py
import scipy
import torch
from diffusers import MiniMaxMusic3Pipeline

pipe = MiniMaxMusic3Pipeline.from_pretrained("MiniMaxAI/MiniMax-Music3", torch_dtype=torch.bfloat16)
pipe = pipe.to("cuda")

lyrics = """[verse]
Morning light filtering through the pine
Every quiet street is yours and mine
[chorus]
Softly the world begins to breathe"""

prompt = (
    "A warm acoustic pop song with intimate female vocals, fingerpicked guitar, soft piano, "
    "and a gradual emotional build into a wide final chorus."
)

audio = pipe(
    prompt=prompt,
    lyrics=lyrics,
    audio_duration=60.0,
    generator=torch.Generator("cuda").manual_seed(7),
).audios[0]

scipy.io.wavfile.write("minimax_music3.wav", rate=pipe.sampling_rate, data=audio.T)
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
- The pipeline returns the vocoder's native 44.1 kHz stereo output. The reference server additionally resamples to 32
  kHz; apply your own resampling if you need that exact rate.

## MiniMaxMusic3Pipeline

[[autodoc]] MiniMaxMusic3Pipeline
	- all
	- __call__

## MiniMaxMusic3ConditionEncoder

[[autodoc]] MiniMaxMusic3ConditionEncoder

## MiniMaxMusic3RVQDepthDecoder

[[autodoc]] MiniMaxMusic3RVQDepthDecoder

## MiniMaxMusic3Vocoder

[[autodoc]] MiniMaxMusic3Vocoder
