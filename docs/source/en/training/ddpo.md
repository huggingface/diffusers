<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Reinforcement learning training with DDPO

It is possible to leverage the 🤗 `trl` library along with `diffusers` to fine-tune Stable Diffusion on a reward function via reinforcement learning. This is done via an algorithm called Denoising Diffusion Policy Optimization (DDPO), introduced by Black et al. in [Training Diffusion Models with Reinforcement Learning](https://arxiv.org/abs/2305.13301). 🤗 `trl` implements a dedicated trainer class for this `DDPOTrainer`.

## Resources

* `DDPOTrainer` [documentation](https://huggingface.co/docs/trl/ddpo_trainer)
* Supplementary [blog post](https://huggingface.co/blog/trl-ddpo)