<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Hybrid Inference

**Empowering local AI builders with Hybrid Inference**


> [!TIP]
> Hybrid Inference is an [experimental feature](https://huggingface.co/blog/remote_vae).
> Feedback can be provided [here](https://github.com/huggingface/diffusers/issues/new?template=remote-vae-pilot-feedback.yml).



## Why use Hybrid Inference?

Hybrid Inference offers a fast and simple way to offload local generation requirements.

- 🚀 **Reduced Requirements:** Access powerful models without expensive hardware.
- 💎 **Without Compromise:** Achieve the highest quality without sacrificing performance.
- 💰 **Cost Effective:** It's free! 🤑
- 🎯 **Diverse Use Cases:** Fully compatible with Diffusers 🧨 and the wider community.
- 🔧 **Developer-Friendly:** Simple requests, fast responses.

---

## Available Models

* **VAE Decode 🖼️:** Quickly decode latent representations into high-quality images without compromising performance or workflow speed.
* **VAE Encode 🔢:** Efficiently encode images into latent representations for generation and training.
* **Text Encoders 📃 (coming soon):** Compute text embeddings for your prompts quickly and accurately, ensuring a smooth and high-quality workflow.

---

## Integrations

* **[SD.Next](https://github.com/vladmandic/sdnext):** All-in-one UI with direct supports Hybrid Inference.
* **[ComfyUI-HFRemoteVae](https://github.com/kijai/ComfyUI-HFRemoteVae):** ComfyUI node for Hybrid Inference.

## Changelog

- March 10 2025: Added VAE encode
- March 2 2025: Initial release with VAE decoding

## Contents

The documentation is organized into three sections:

* **VAE Decode** Learn the basics of how to use VAE Decode with Hybrid Inference.
* **VAE Encode** Learn the basics of how to use VAE Encode with Hybrid Inference.
* **API Reference** Dive into task-specific settings and parameters.
