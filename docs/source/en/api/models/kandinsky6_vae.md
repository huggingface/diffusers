<!--Copyright 2026 The Kandinsky Team and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Kandinsky 6 VAEs

Kandinsky 6 uses a causal 3D K-VAE for video super-resolution and the MMAudio mel-spectrogram VAE, paired with a
separate BigVGAN [`MMAudioVocoder`], for synchronized audio generation.

## Kandinsky6SRVAE

The causal 3D K-VAE used by [`Kandinsky6SRPipeline`]. It processes arbitrarily long videos in bounded-memory
segments while reproducing the exact output of a single, non-segmented pass.

```python
import torch
from diffusers import Kandinsky6SRVAE

vae = Kandinsky6SRVAE.from_pretrained(
    "kandinskylab/Kandinsky-6.0-VSR-5s-Diffusers", subfolder="vae", torch_dtype=torch.bfloat16
)
```

[[autodoc]] Kandinsky6SRVAE
  - encode
  - decode
  - all

## MMAudioVAE

The mel-spectrogram VAE used by [`Kandinsky6TI2VAPipeline`] when `sample_audio=True`. Its `decode` output is a mel
spectrogram; pass it through [`MMAudioVocoder`] to get a waveform.

The reference implementation can be found at [hkchengrex/MMAudio](https://github.com/hkchengrex/MMAudio) (MIT
license).

```python
import torch
from diffusers import MMAudioVAE

audio_vae = MMAudioVAE.from_pretrained(
    "kandinskylab/Kandinsky-6.0-Pro-distill-5s-Diffusers", subfolder="audio_vae", torch_dtype=torch.bfloat16
)
```

[[autodoc]] MMAudioVAE
  - encode
  - decode
  - all

## MMAudioVocoder

Adapted from the BigVGAN-v2 vocoder MMAudio bundles, itself from
[NVIDIA/BigVGAN](https://github.com/NVIDIA/BigVGAN) (MIT license), with the anti-aliased Snake activations of
[alias-free-torch](https://github.com/junjun3518/alias-free-torch) (Apache License 2.0).

[[autodoc]] MMAudioVocoder
  - forward
