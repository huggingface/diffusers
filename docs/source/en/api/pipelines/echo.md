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

[Echo](https://github.com/jd-opensource/JoyAI-Echo) is a long-video generation model. It adds an optional clean first
frame, ordered image/audio memory slots, and a stochastic few-step Distribution Matching Distillation (DMD) sampler.
The pipeline generates synchronized video and audio.

Echo is implemented as a Modular Pipeline so its text encoding, memory conditioning, stochastic DMD denoising, and
decoding blocks can be run as a complete workflow or composed independently.

## Convert the checkpoint

Convert the BF16 Echo release checkpoint before loading it. The converter reuses the Gemma text encoder and
tokenizer directly from `google/gemma-3-12b-it`.

```bash
python scripts/convert_echo_to_diffusers.py \
  --checkpoint /path/to/echo15_full_dmd \
  --output-path /path/to/Echo-Diffusers \
  --repo-id jdopensource/JoyAI-Echo
```

The Gemma repository is gated, so users must accept its license and authenticate with Hugging Face before loading the
pipeline. Pass a different `--base-model` only when the compatible Gemma model and tokenizer are stored together at
that repository or path root. `--repo-id` records portable Hub references for the converted Echo components; without
it, the index targets the local output path.

## Inference

The released model uses 241 frames in its long-video example. The video RoPE coordinates remain at the training rate
of 24 fps, independently of the output container rate.

```py
import torch
import torchaudio
from PIL import Image

from diffusers import ComponentsManager, ModularPipeline
from diffusers.utils import encode_video


model_path = "/path/to/Echo-Diffusers"
manager = ComponentsManager()
pipe = ModularPipeline.from_pretrained(model_path, components_manager=manager)
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

The default DMD sigma schedule is the released eight-step schedule. It predicts x0 at every step and re-noises with
fresh Gaussian noise at the next sigma, so a seeded `torch.Generator` controls both the initial noise and all
intermediate re-noising.

Raw audio-memory encoding requires `torchaudio`. For reference parity, keep `audio_vae` in FP32 as shown above.
Modular workflows can cache and reuse the condition encoder's packed token outputs by running that block separately.

## EchoModularPipeline

[[autodoc]] EchoModularPipeline

## EchoBlocks

[[autodoc]] EchoBlocks
