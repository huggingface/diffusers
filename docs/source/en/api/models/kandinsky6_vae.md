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
separate BigVGAN [`~pipelines.kandinsky6.MMAudioVocoder`], for synchronized audio generation.

## Kandinsky6SRVAE

[[autodoc]] Kandinsky6SRVAE
  - encode
  - decode
  - all

## MMAudioVAE

[[autodoc]] MMAudioVAE
  - encode
  - decode
  - all

## MMAudioVocoder

[[autodoc]] pipelines.kandinsky6.MMAudioVocoder
  - forward
