<!-- Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License. -->

# Echo

[Echo](https://github.com/jd-opensource/JoyAI-Echo) is a long-video generation model. It supports an optional clean
first frame, ordered image/audio memory slots, and a stochastic few-step Distribution Matching Distillation (DMD)
sampler.
The pipeline generates synchronized video and audio.

Echo is implemented as a [`ModularPipeline`] so its text encoding, memory conditioning, stochastic DMD denoising, and
decoding blocks can be run as a complete workflow or composed independently.

## Inference

Load the official [jdopensource/JoyAI-Echo](https://huggingface.co/jdopensource/JoyAI-Echo) checkpoint directly.
It uses the text encoder and tokenizer from [google/gemma-3-12b-it](https://huggingface.co/google/gemma-3-12b-it).
The Gemma repository is gated, so accept its license and authenticate with Hugging Face before loading the pipeline.

The released model uses 241 frames in its long-video example. The video RoPE coordinates remain at the training rate
of 24 fps, independently of the output container rate.

```py
import torch
import torchaudio
from PIL import Image

from diffusers import ComponentsManager, ModularPipeline
from diffusers.utils import encode_video

manager = ComponentsManager()
pipe = ModularPipeline.from_pretrained("jdopensource/JoyAI-Echo", components_manager=manager)
pipe.load_components(dtype={"default": torch.bfloat16, "audio_vae": torch.float32})
manager.enable_auto_cpu_offload(device="cuda")
pipe.vae.enable_tiling()

first_frame = Image.open("first_frame.png").convert("RGB")
memory_images = [Image.open(path).convert("RGB") for path in ["memory_0.png", "memory_1.png"]]
memory_audio_with_rates = [torchaudio.load(path) for path in ["memory_0.wav", "memory_1.wav"]]
memory_audio = [waveform for waveform, _ in memory_audio_with_rates]
memory_audio_rates = [sample_rate for _, sample_rate in memory_audio_with_rates]

output = pipe(
    prompt="A cinematic dialogue scene in a quiet cafe.",
    image=first_frame,
    memory_images=memory_images,
    memory_audio_waveforms=memory_audio,
    memory_audio_sample_rates=memory_audio_rates,
    width=1280,
    height=736,
    num_frames=241,
    frame_rate=25.0,
    model_frame_rate=24.0,
    generator=torch.Generator(device="cuda").manual_seed(42),
    output_type="np",
    output=["videos", "audio"],
)

encode_video(
    output["videos"][0],
    fps=25,
    audio=output["audio"][0].float().cpu(),
    audio_sample_rate=pipe.vocoder.config.output_sampling_rate,
    output_path="echo.mp4",
)
```

The default DMD sigma schedule is the released eight-step schedule. It predicts `x0` at every step and re-noises with
fresh Gaussian noise at the next sigma, so a seeded `torch.Generator` controls both the initial noise and all
intermediate re-noising.

Raw audio-memory encoding requires `torchaudio`. For reference parity, keep `audio_vae` in FP32 as shown above.
Modular workflows can cache the VAE encoder's normalized, unpacked tensors. Video latents have shape
`(batch, channels, frames, height, width)` and audio latents have shape `(batch, channels, time, mel_bins)`.
The core `denoise` block packs those tensors, expands conditioning for `num_videos_per_prompt`, and unpacks
denoised outputs back to the same VAE form. Pass initial `latents` and `audio_latents` in that unpacked form too.
Decoders take normalized VAE tensors and denormalize immediately before
decoding. `output=["latents", "audio_latents"]` returns normalized VAE tensors. With `output_type="latent"` and
`output=["videos", "audio"]`, you get the denormalized VAE tensors without decoding.

## EchoModularPipeline

[[autodoc]] EchoModularPipeline

## EchoBlocks

[[autodoc]] EchoBlocks
