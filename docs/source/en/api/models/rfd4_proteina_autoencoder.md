<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# RFD4ProteinaAutoencoder

The autoencoder of [RFD4-Proteína](../pipelines/rfd4_proteina). `encode` maps atom37 coordinates to per-token local
latents, and `decode` turns latents and backbone positions back into atom37 structure and residue identities. It is
registered in the pipeline as `vae`.

```py
from diffusers import RFD4ProteinaAutoencoder

vae = RFD4ProteinaAutoencoder.from_pretrained("nvidia/RFD4-Proteina-Diffusers", subfolder="vae")
```

## RFD4ProteinaAutoencoder

[[autodoc]] RFD4ProteinaAutoencoder
  - encode
  - decode
  - forward

## RFD4ProteinaEncoderOutput

[[autodoc]] models.autoencoders.autoencoder_rfd4_proteina.RFD4ProteinaEncoderOutput

## RFD4ProteinaDecoderOutput

[[autodoc]] models.autoencoders.autoencoder_rfd4_proteina.RFD4ProteinaDecoderOutput
