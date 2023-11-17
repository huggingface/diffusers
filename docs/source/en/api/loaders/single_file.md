<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Single files

Diffusers supports loading pretrained pipeline (or model) weights stored in a single file, such as a `ckpt` or `safetensors` file. These single file types are typically produced from community trained models. There are three classes for loading single file weights:

- [`FromSingleFileMixin`] supports loading pretrained pipeline weights stored in a single file, which can either be a `ckpt` or `safetensors` file.
- [`FromOriginalVAEMixin`] supports loading a pretrained [`AutoencoderKL`] from pretrained ControlNet weights stored in a single file, which can either be a `ckpt` or `safetensors` file.
- [`FromOriginalControlnetMixin`] supports loading pretrained ControlNet weights stored in a single file, which can either be a `ckpt` or `safetensors` file.

<Tip>

To learn more about how to load single file weights, see the [Load different Stable Diffusion formats](../../using-diffusers/other-formats) loading guide.

</Tip>

## FromSingleFileMixin

[[autodoc]] loaders.single_file.FromSingleFileMixin

## FromOriginalVAEMixin

[[autodoc]] loaders.single_file.FromOriginalVAEMixin

## FromOriginalControlnetMixin

[[autodoc]] loaders.single_file.FromOriginalControlnetMixin