<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# PEFT

Diffusers supports working with adapters (such as [LoRA](../../using-diffusers/loading_adapters)) via the [`peft` library](https://huggingface.co/docs/peft/index). We provide a `PeftAdapterMixin` class to handle this for modeling classes in Diffusers (such as [`UNet2DConditionModel`]).

<Tip>

Refer to [this doc](../../tutorials/using_peft_for_inference.md) to get an overview of how to work with `peft` in Diffusers for inference.

</Tip>

## PeftAdapterMixin

[[autodoc]] loaders.peft.PeftAdapterMixin