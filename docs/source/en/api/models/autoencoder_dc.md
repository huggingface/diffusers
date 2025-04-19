<!-- Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License. -->

# AutoencoderDC

The 2D Autoencoder model used in [SANA](https://huggingface.co/papers/2410.10629) and introduced in [DCAE](https://huggingface.co/papers/2410.10733) by authors Junyu Chen\*, Han Cai\*, Junsong Chen, Enze Xie, Shang Yang, Haotian Tang, Muyang Li, Yao Lu, Song Han from MIT HAN Lab.

The abstract from the paper is:

*We present Deep Compression Autoencoder (DC-AE), a new family of autoencoder models for accelerating high-resolution diffusion models. Existing autoencoder models have demonstrated impressive results at a moderate spatial compression ratio (e.g., 8x), but fail to maintain satisfactory reconstruction accuracy for high spatial compression ratios (e.g., 64x). We address this challenge by introducing two key techniques: (1) Residual Autoencoding, where we design our models to learn residuals based on the space-to-channel transformed features to alleviate the optimization difficulty of high spatial-compression autoencoders; (2) Decoupled High-Resolution Adaptation, an efficient decoupled three-phases training strategy for mitigating the generalization penalty of high spatial-compression autoencoders. With these designs, we improve the autoencoder's spatial compression ratio up to 128 while maintaining the reconstruction quality. Applying our DC-AE to latent diffusion models, we achieve significant speedup without accuracy drop. For example, on ImageNet 512x512, our DC-AE provides 19.1x inference speedup and 17.9x training speedup on H100 GPU for UViT-H while achieving a better FID, compared with the widely used SD-VAE-f8 autoencoder. Our code is available at [this https URL](https://github.com/mit-han-lab/efficientvit).*

The following DCAE models are released and supported in Diffusers.

| Diffusers format | Original format |
|:----------------:|:---------------:|
| [`mit-han-lab/dc-ae-f32c32-sana-1.0-diffusers`](https://huggingface.co/mit-han-lab/dc-ae-f32c32-sana-1.0-diffusers) | [`mit-han-lab/dc-ae-f32c32-sana-1.0`](https://huggingface.co/mit-han-lab/dc-ae-f32c32-sana-1.0)
| [`mit-han-lab/dc-ae-f32c32-in-1.0-diffusers`](https://huggingface.co/mit-han-lab/dc-ae-f32c32-in-1.0-diffusers) | [`mit-han-lab/dc-ae-f32c32-in-1.0`](https://huggingface.co/mit-han-lab/dc-ae-f32c32-in-1.0)
| [`mit-han-lab/dc-ae-f32c32-mix-1.0-diffusers`](https://huggingface.co/mit-han-lab/dc-ae-f32c32-mix-1.0-diffusers) | [`mit-han-lab/dc-ae-f32c32-mix-1.0`](https://huggingface.co/mit-han-lab/dc-ae-f32c32-mix-1.0)
| [`mit-han-lab/dc-ae-f64c128-in-1.0-diffusers`](https://huggingface.co/mit-han-lab/dc-ae-f64c128-in-1.0-diffusers) | [`mit-han-lab/dc-ae-f64c128-in-1.0`](https://huggingface.co/mit-han-lab/dc-ae-f64c128-in-1.0)
| [`mit-han-lab/dc-ae-f64c128-mix-1.0-diffusers`](https://huggingface.co/mit-han-lab/dc-ae-f64c128-mix-1.0-diffusers) | [`mit-han-lab/dc-ae-f64c128-mix-1.0`](https://huggingface.co/mit-han-lab/dc-ae-f64c128-mix-1.0)
| [`mit-han-lab/dc-ae-f128c512-in-1.0-diffusers`](https://huggingface.co/mit-han-lab/dc-ae-f128c512-in-1.0-diffusers) | [`mit-han-lab/dc-ae-f128c512-in-1.0`](https://huggingface.co/mit-han-lab/dc-ae-f128c512-in-1.0)
| [`mit-han-lab/dc-ae-f128c512-mix-1.0-diffusers`](https://huggingface.co/mit-han-lab/dc-ae-f128c512-mix-1.0-diffusers) | [`mit-han-lab/dc-ae-f128c512-mix-1.0`](https://huggingface.co/mit-han-lab/dc-ae-f128c512-mix-1.0)

This model was contributed by [lawrence-cj](https://github.com/lawrence-cj).

Load a model in Diffusers format with [`~ModelMixin.from_pretrained`].

```python
from diffusers import AutoencoderDC

ae = AutoencoderDC.from_pretrained("mit-han-lab/dc-ae-f32c32-sana-1.0-diffusers", torch_dtype=torch.float32).to("cuda")
```

## Load a model in Diffusers via `from_single_file`

```python
from difusers import AutoencoderDC

ckpt_path = "https://huggingface.co/mit-han-lab/dc-ae-f32c32-sana-1.0/blob/main/model.safetensors"
model = AutoencoderDC.from_single_file(ckpt_path) 

```

The `AutoencoderDC` model has `in` and `mix` single file checkpoint variants that have matching checkpoint keys, but use different scaling factors. It is not possible for Diffusers to automatically infer the correct config file to use with the model based on just the checkpoint and will default to configuring the model using the `mix` variant config file. To override the automatically determined config, please use the `config` argument when using single file loading with `in` variant checkpoints. 

```python
from diffusers import AutoencoderDC

ckpt_path = "https://huggingface.co/mit-han-lab/dc-ae-f128c512-in-1.0/blob/main/model.safetensors"
model = AutoencoderDC.from_single_file(ckpt_path, config="mit-han-lab/dc-ae-f128c512-in-1.0-diffusers")
```


## AutoencoderDC

[[autodoc]] AutoencoderDC
  - encode
  - decode
  - all

## DecoderOutput

[[autodoc]] models.autoencoders.vae.DecoderOutput

