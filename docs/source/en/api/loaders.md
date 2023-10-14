<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Loaders

Adapters (textual inversion, LoRA, hypernetworks) allow you to modify a diffusion model to generate images in a specific style without training or finetuning the entire model. The adapter weights are typically only a tiny fraction of the pretrained model's which making them very portable. 🤗 Diffusers provides an easy-to-use `LoaderMixin` API to load adapter weights.

<Tip warning={true}>

🧪 The `LoaderMixins` are highly experimental and prone to future changes. To use private or [gated](https://huggingface.co/docs/hub/models-gated#gated-models) models, log-in with `huggingface-cli login`.

</Tip>

## UNet2DConditionLoadersMixin

[[autodoc]] loaders.UNet2DConditionLoadersMixin

## TextualInversionLoaderMixin

[[autodoc]] loaders.TextualInversionLoaderMixin

## StableDiffusionXLLoraLoaderMixin

[[autodoc]] loaders.StableDiffusionXLLoraLoaderMixin

## LoraLoaderMixin

[[autodoc]] loaders.LoraLoaderMixin

## FromSingleFileMixin

[[autodoc]] loaders.FromSingleFileMixin

## FromOriginalControlnetMixin

[[autodoc]] loaders.FromOriginalControlnetMixin

## FromOriginalVAEMixin

[[autodoc]] loaders.FromOriginalVAEMixin
