<!-- Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License. -->

# AutoencoderDC

*The 2D Autoencoder model used in [SANA](https://huggingface.co/papers/2410.10629) and introduced in [DCAE](https://huggingface.co/papers/2410.10733) by authors Junyu Chen\*, Han Cai\*, Junsong Chen, Enze Xie, Shang Yang, Haotian Tang, Muyang Li, Yao Lu, Song Han from MIT HAN Lab.*

The following DCAE models are released and supported in Diffusers:
- [`mit-han-lab/dc-ae-f32c32-sana-1.0`](https://huggingface.co/mit-han-lab/dc-ae-f32c32-sana-1.0)
- [`mit-han-lab/dc-ae-f32c32-in-1.0`](https://huggingface.co/mit-han-lab/dc-ae-f32c32-in-1.0)
- [`mit-han-lab/dc-ae-f32c32-mix-1.0`](https://huggingface.co/mit-han-lab/dc-ae-f32c32-mix-1.0)
- [`mit-han-lab/dc-ae-f64c128-in-1.0`](https://huggingface.co/mit-han-lab/dc-ae-f64c128-in-1.0)
- [`mit-han-lab/dc-ae-f64c128-mix-1.0`](https://huggingface.co/mit-han-lab/dc-ae-f64c128-mix-1.0)
- [`mit-han-lab/dc-ae-f128c512-in-1.0`](https://huggingface.co/mit-han-lab/dc-ae-f128c512-in-1.0)
- [`mit-han-lab/dc-ae-f128c512-mix-1.0`](https://huggingface.co/mit-han-lab/dc-ae-f128c512-mix-1.0)

The models can be loaded with the following code snippet.

```python
from diffusers import AutoencoderDC

ae = AutoencoderDC.from_pretrained("mit-han-lab/dc-ae-f32c32-sana-1.0-diffusers", torch_dtype=torch.float32).to("cuda")
```

## Single file loading

The `AutoencoderDC` implementation supports loading checkpoints shipped in the original format by MIT HAN Lab. The following example demonstrates how to load the `f128c512` checkpoint:

```python
from diffusers import AutoencoderDC

model_name = "dc-ae-f128c512-mix-1.0"
ae = AutoencoderDC.from_single_file(
    f"https://huggingface.co/mit-han-lab/{model_name}/model.safetensors",
    original_config=f"https://huggingface.co/mit-han-lab/{model_name}/resolve/main/config.json"
)
```

## AutoencoderDC

[[autodoc]] AutoencoderDC
    - decode
    - all

## DecoderOutput

[[autodoc]] models.autoencoders.vae.DecoderOutput

