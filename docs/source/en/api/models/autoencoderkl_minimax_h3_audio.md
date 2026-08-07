<!-- Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License. -->

# AutoencoderKLMiniMaxH3Audio

The audio autoencoder used in [MiniMax-H3](https://huggingface.co/MiniMaxAI) by MiniMax. It is waveform in and waveform out, with no mel front-end and no separate vocoder: a DAC-lineage strided convolutional encoder, a causal-attention projection onto the diffusion latent width, and a BigVGAN decoder.

The encoder hops 800 samples at 32 kHz, i.e. 40 latents per second, so a waveform of `800 * n` samples encodes to `n` latents. Waveforms that are not a whole number of hops are right-padded.

The causal-attention projection goes through the attention dispatcher, so `set_attention_backend` applies to it; its mask is `is_causal=True`, which every backend honours except `_native_npu`, whose kernel takes no causal flag.

The autoencoder is **mono**, and it normalizes latents per channel with `latents_mean` / `latents_std` rather than a scalar `scaling_factor`. MiniMax-H3 carries stereo as two *batch* items, and it always consumes the posterior mean (`latent_dist.mode()`), never a sample.

```python
import torch
from diffusers.utils.torch_utils import get_device
from diffusers import AutoencoderKLMiniMaxH3Audio


device = get_device()
audio_vae = AutoencoderKLMiniMaxH3Audio.from_pretrained(
    "MiniMaxAI/MiniMax-H3", subfolder="audio_vae", dtype=torch.float32
).to(device)
```

## AutoencoderKLMiniMaxH3Audio

[[autodoc]] AutoencoderKLMiniMaxH3Audio
    - encode
    - decode
    - all
