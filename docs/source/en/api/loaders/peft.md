<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# PEFT

Diffusers supports loading adapters such as [LoRA](../../using-diffusers/loading_adapters) with the [PEFT](https://huggingface.co/docs/peft/index) library with the [`~loaders.peft.PeftAdapterMixin`] class. This allows modeling classes in Diffusers like [`UNet2DConditionModel`], [`SD3Transformer2DModel`] to operate with an adapter.

<Tip>

Refer to the [Inference with PEFT](../../tutorials/using_peft_for_inference.md) tutorial for an overview of how to use PEFT in Diffusers for inference.

</Tip>

## PeftAdapterMixin

[[autodoc]] loaders.peft.PeftAdapterMixin
