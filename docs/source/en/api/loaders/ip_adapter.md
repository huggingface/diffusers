<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# IP-Adapter

[IP-Adapter](https://hf.co/papers/2308.06721) is a lightweight adapter that enables prompting a diffusion model with an image. This method decouples the cross-attention layers of the image and text features. The image features are generated from an image encoder.

<Tip>

Learn how to load an IP-Adapter checkpoint and image in the IP-Adapter [loading](../../using-diffusers/loading_adapters#ip-adapter) guide, and you can see how to use it in the [usage](../../using-diffusers/ip_adapter) guide.

</Tip>

## IPAdapterMixin

[[autodoc]] loaders.ip_adapter.IPAdapterMixin

## IPAdapterMaskProcessor

[[autodoc]] image_processor.IPAdapterMaskProcessor