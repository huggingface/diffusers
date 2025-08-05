<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Reinforcement learning training with DDPO

You can fine-tune Stable Diffusion on a reward function via reinforcement learning with the 🤗 TRL library and 🤗 Diffusers. This is done with the Denoising Diffusion Policy Optimization (DDPO) algorithm introduced by Black et al. in [Training Diffusion Models with Reinforcement Learning](https://huggingface.co/papers/2305.13301), which is implemented in 🤗 TRL with the [`~trl.DDPOTrainer`].

For more information, check out the [`~trl.DDPOTrainer`] API reference and the [Finetune Stable Diffusion Models with DDPO via TRL](https://huggingface.co/blog/trl-ddpo) blog post.