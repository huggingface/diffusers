<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# UNet

Some training methods - like LoRA and Custom Diffusion - typically target the UNet's attention layers, but these training methods can also target other non-attention layers. Instead of training all of a model's parameters, only a subset of the parameters are trained, which is faster and more efficient. This class is useful if you're *only* loading weights into a UNet. If you need to load weights into the text encoder or a text encoder and UNet, try using the [`~loaders.StableDiffusionLoraLoaderMixin.load_lora_weights`] function instead.

The [`UNet2DConditionLoadersMixin`] class provides functions for loading and saving weights, fusing and unfusing LoRAs, disabling and enabling LoRAs, and setting and deleting adapters.

<Tip>

To learn more about how to load LoRA weights, see the [LoRA](../../using-diffusers/loading_adapters#lora) loading guide.

</Tip>

## UNet2DConditionLoadersMixin

[[autodoc]] loaders.unet.UNet2DConditionLoadersMixin