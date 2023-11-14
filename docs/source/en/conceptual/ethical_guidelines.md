<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# 🧨 Diffusers’ Ethical Guidelines

## Preamble

[Diffusers](https://huggingface.co/docs/diffusers/index) provides pre-trained diffusion models and serves as a modular toolbox for inference and training.

Given its real case applications in the world and potential negative impacts on society, we think it is important to provide the project with ethical guidelines to guide the development, users’ contributions, and usage of the Diffusers library.

The risks associated with using this technology are still being examined, but to name a few: copyrights issues for artists; deep-fake exploitation; sexual content generation in inappropriate contexts; non-consensual impersonation; harmful social biases perpetuating the oppression of marginalized groups.
We will keep tracking risks and adapt the following guidelines based on the community's responsiveness and valuable feedback.


## Scope

The Diffusers community will apply the following ethical guidelines to the project’s development and help coordinate how the community will integrate the contributions, especially concerning sensitive topics related to ethical concerns.


## Ethical guidelines

The following ethical guidelines apply generally, but we will primarily implement them when dealing with ethically sensitive issues while making a technical choice. Furthermore, we commit to adapting those ethical principles over time following emerging harms related to the state of the art of the technology in question.

- **Transparency**: we are committed to being transparent in managing PRs, explaining our choices to users, and making technical decisions.

- **Consistency**: we are committed to guaranteeing our users the same level of attention in project management, keeping it technically stable and consistent.

- **Simplicity**: with a desire to make it easy to use and exploit the Diffusers library, we are committed to keeping the project’s goals lean and coherent.

- **Accessibility**: the Diffusers project helps lower the entry bar for contributors who can help run it even without technical expertise. Doing so makes research artifacts more accessible to the community.

- **Reproducibility**: we aim to be transparent about the reproducibility of upstream code, models, and datasets when made available through the Diffusers library.

- **Responsibility**: as a community and through teamwork, we hold a collective responsibility to our users by anticipating and mitigating this technology's potential risks and dangers.


## Examples of implementations: Safety features and Mechanisms

The team works daily to make the technical and non-technical tools available to deal with the potential ethical and social risks associated with diffusion technology. Moreover, the community's input is invaluable in ensuring these features' implementation and raising awareness with us.

- [**Community tab**](https://huggingface.co/docs/hub/repositories-pull-requests-discussions): it enables the community to discuss and better collaborate on a project.

- **Bias exploration and evaluation**: the Hugging Face team provides a [space](https://huggingface.co/spaces/society-ethics/DiffusionBiasExplorer) to demonstrate the biases in Stable Diffusion interactively. In this sense, we support and encourage bias explorers and evaluations.

- **Encouraging safety in deployment**

  - [**Safe Stable Diffusion**](https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_diffusion/stable_diffusion_safe): It mitigates the well-known issue that models, like Stable Diffusion, that are trained on unfiltered, web-crawled datasets tend to suffer from inappropriate degeneration. Related paper: [Safe Latent Diffusion: Mitigating Inappropriate Degeneration in Diffusion Models](https://arxiv.org/abs/2211.05105).

  - [**Safety Checker**](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/safety_checker.py): It checks and compares the class probability of a set of hard-coded harmful concepts in the embedding space against an image after it has been generated. The harmful concepts are intentionally hidden to prevent reverse engineering of the checker.

- **Staged released on the Hub**: in particularly sensitive situations, access to some repositories should be restricted. This staged release is an intermediary step that allows the repository’s authors to have more control over its use.

- **Licensing**: [OpenRAILs](https://huggingface.co/blog/open_rail), a new type of licensing, allow us to ensure free access while having a set of restrictions that ensure more responsible use.
