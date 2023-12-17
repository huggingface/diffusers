<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# LoRA

LoRA is a fast and lightweight training method that inserts and trains a significantly smaller number of parameters instead of all the model parameters. This produces a smaller file (~100 MBs) and makes it easier to quickly train a model to learn a new concept. LoRA weights are typically loaded into the UNet, text encoder or both. There are two classes for loading LoRA weights:

- [`LoraLoaderMixin`] provides functions for loading and unloading, fusing and unfusing, enabling and disabling, and more functions for managing LoRA weights. This class can be used with any model.
- [`StableDiffusionXLLoraLoaderMixin`] is a [Stable Diffusion (SDXL)](../../api/pipelines/stable_diffusion/stable_diffusion_xl) version of the [`LoraLoaderMixin`] class for loading and saving LoRA weights. It can only be used with the SDXL model.

<Tip>

To learn more about how to load LoRA weights, see the [LoRA](../../using-diffusers/loading_adapters#lora) loading guide.

</Tip>

## LoraLoaderMixin

[[autodoc]] loaders.lora.LoraLoaderMixin

## StableDiffusionXLLoraLoaderMixin

[[autodoc]] loaders.lora.StableDiffusionXLLoraLoaderMixin