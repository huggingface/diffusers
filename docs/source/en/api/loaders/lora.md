<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# LoRA

LoRA is a fast and lightweight training method that inserts and trains a significantly smaller number of parameters instead of all the model parameters. This produces a smaller file (~100 MBs) and makes it easier to quickly train a model to learn a new concept. LoRA weights are typically loaded into the denoiser, text encoder or both. The denoiser usually corresponds to a UNet ([`UNet2DConditionModel`], for example) or a Transformer ([`SD3Transformer2DModel`], for example). There are several classes for loading LoRA weights:

- [`StableDiffusionLoraLoaderMixin`] provides functions for loading and unloading, fusing and unfusing, enabling and disabling, and more functions for managing LoRA weights. This class can be used with any model.
- [`StableDiffusionXLLoraLoaderMixin`] is a [Stable Diffusion (SDXL)](../../api/pipelines/stable_diffusion/stable_diffusion_xl) version of the [`StableDiffusionLoraLoaderMixin`] class for loading and saving LoRA weights. It can only be used with the SDXL model.
- [`SD3LoraLoaderMixin`] provides similar functions for [Stable Diffusion 3](https://huggingface.co/blog/sd3).
- [`FluxLoraLoaderMixin`] provides similar functions for [Flux](https://huggingface.co/docs/diffusers/main/en/api/pipelines/flux).
- [`CogVideoXLoraLoaderMixin`] provides similar functions for [CogVideoX](https://huggingface.co/docs/diffusers/main/en/api/pipelines/cogvideox).
- [`Mochi1LoraLoaderMixin`] provides similar functions for [Mochi](https://huggingface.co/docs/diffusers/main/en/api/pipelines/mochi).
- [`AuraFlowLoraLoaderMixin`] provides similar functions for [AuraFlow](https://huggingface.co/fal/AuraFlow).
- [`LTXVideoLoraLoaderMixin`] provides similar functions for [LTX-Video](https://huggingface.co/docs/diffusers/main/en/api/pipelines/ltx_video).
- [`SanaLoraLoaderMixin`] provides similar functions for [Sana](https://huggingface.co/docs/diffusers/main/en/api/pipelines/sana).
- [`HunyuanVideoLoraLoaderMixin`] provides similar functions for [HunyuanVideo](https://huggingface.co/docs/diffusers/main/en/api/pipelines/hunyuan_video).
- [`Lumina2LoraLoaderMixin`] provides similar functions for [Lumina2](https://huggingface.co/docs/diffusers/main/en/api/pipelines/lumina2).
- [`WanLoraLoaderMixin`] provides similar functions for [Wan](https://huggingface.co/docs/diffusers/main/en/api/pipelines/wan).
- [`SkyReelsV2LoraLoaderMixin`] provides similar functions for [SkyReels-V2](https://huggingface.co/docs/diffusers/main/en/api/pipelines/skyreels_v2).
- [`CogView4LoraLoaderMixin`] provides similar functions for [CogView4](https://huggingface.co/docs/diffusers/main/en/api/pipelines/cogview4).
- [`AmusedLoraLoaderMixin`] is for the [`AmusedPipeline`].
- [`HiDreamImageLoraLoaderMixin`] provides similar functions for [HiDream Image](https://huggingface.co/docs/diffusers/main/en/api/pipelines/hidream)
- [`QwenImageLoraLoaderMixin`] provides similar functions for [Qwen Image](https://huggingface.co/docs/diffusers/main/en/api/pipelines/qwen)
- [`LoraBaseMixin`] provides a base class with several utility methods to fuse, unfuse, unload, LoRAs and more.

> [!TIP]
> To learn more about how to load LoRA weights, see the [LoRA](../../tutorials/using_peft_for_inference) loading guide.

## LoraBaseMixin

[[autodoc]] loaders.lora_base.LoraBaseMixin

## StableDiffusionLoraLoaderMixin

[[autodoc]] loaders.lora_pipeline.StableDiffusionLoraLoaderMixin

## StableDiffusionXLLoraLoaderMixin

[[autodoc]] loaders.lora_pipeline.StableDiffusionXLLoraLoaderMixin

## SD3LoraLoaderMixin

[[autodoc]] loaders.lora_pipeline.SD3LoraLoaderMixin

## FluxLoraLoaderMixin

[[autodoc]] loaders.lora_pipeline.FluxLoraLoaderMixin

## CogVideoXLoraLoaderMixin

[[autodoc]] loaders.lora_pipeline.CogVideoXLoraLoaderMixin

## Mochi1LoraLoaderMixin

[[autodoc]] loaders.lora_pipeline.Mochi1LoraLoaderMixin
## AuraFlowLoraLoaderMixin

[[autodoc]] loaders.lora_pipeline.AuraFlowLoraLoaderMixin

## LTXVideoLoraLoaderMixin

[[autodoc]] loaders.lora_pipeline.LTXVideoLoraLoaderMixin

## SanaLoraLoaderMixin

[[autodoc]] loaders.lora_pipeline.SanaLoraLoaderMixin

## HunyuanVideoLoraLoaderMixin

[[autodoc]] loaders.lora_pipeline.HunyuanVideoLoraLoaderMixin

## Lumina2LoraLoaderMixin

[[autodoc]] loaders.lora_pipeline.Lumina2LoraLoaderMixin

## CogView4LoraLoaderMixin

[[autodoc]] loaders.lora_pipeline.CogView4LoraLoaderMixin

## WanLoraLoaderMixin

[[autodoc]] loaders.lora_pipeline.WanLoraLoaderMixin

## SkyReelsV2LoraLoaderMixin

[[autodoc]] loaders.lora_pipeline.SkyReelsV2LoraLoaderMixin

## AmusedLoraLoaderMixin

[[autodoc]] loaders.lora_pipeline.AmusedLoraLoaderMixin

## HiDreamImageLoraLoaderMixin

[[autodoc]] loaders.lora_pipeline.HiDreamImageLoraLoaderMixin

## QwenImageLoraLoaderMixin

[[autodoc]] loaders.lora_pipeline.QwenImageLoraLoaderMixin

## KandinskyLoraLoaderMixin
[[autodoc]] loaders.lora_pipeline.KandinskyLoraLoaderMixin

## LoraBaseMixin

[[autodoc]] loaders.lora_base.LoraBaseMixin