<!-- Copyright 2026 The Kandinsky Lab Team and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. -->

# AutoencoderKLKVAEAudio

A 1D convolutional variational autoencoder (VAE) with KL loss for audio, introduced by the Kandinsky Lab
Team in [KVAE-Audio](https://huggingface.co/kandinskylab/KVAE-Audio). It compresses raw, full-band (48 kHz)
waveforms into compact continuous latents and reconstructs them with high fidelity across speech, music, and
general audio.

The model can be loaded with the following code snippet.

```python
import torch
from diffusers import AutoencoderKLKVAEAudio

vae = AutoencoderKLKVAEAudio.from_pretrained("kandinskylab/KVAE-Audio", subfolder="diffusers", dtype=torch.float32)
```

## AutoencoderKLKVAEAudio

[[autodoc]] AutoencoderKLKVAEAudio
  - decode
  - encode
  - all
