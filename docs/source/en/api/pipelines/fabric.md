<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# FABRIC

[FABRIC: Personalizing Diffusion Models with Iterative Feedback](https://huggingface.co/papers/2307.10159) (FABRIC) is by Dimitri von Rütte, Elisabetta Fedele, Jonathan Thomm and Lukas Wolf.

FABRIC is a training-free approach that conditions the diffusion process on a set of feedback images, applicable to a wide range of popular diffusion models. It is created by researchers and engineers from [ETH Zürich, Switzerland](https://github.com/sd-fabric). The [`FabricPipeline`] can generate photo-realistic images given any text input using Stable Diffusion.

The abstract from the paper is:

*In an era where visual content generation is increasingly driven by machine learning, the integration of human feedback into generative models presents significant opportunities for enhancing user experience and output quality. This study explores strategies for incorporating iterative human feedback into the generative process of diffusion-based text-to-image models. We propose FABRIC, a training-free approach applicable to a wide range of popular diffusion models, which exploits the self-attention layer present in the most widely used architectures to condition the diffusion process on a set of feedback images. To ensure a rigorous assessment of our approach, we introduce a comprehensive evaluation methodology, offering a robust mechanism to quantify the performance of generative visual models that integrate human feedback. We show that generation results improve over multiple rounds of iterative feedback through exhaustive analysis, implicitly optimizing arbitrary user preferences. The potential applications of these findings extend to fields such as personalized content creation and customization*

The original codebase can be found at [sd-fabric/fabric](https://github.com/sd-fabric/fabric), and available checkpoints are [dreamlike-art/dreamlike-photoreal-2.0](https://huggingface.co/dreamlike-art/dreamlike-photoreal-2.0), [runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5), and [stabilityai/stable-diffusion-2-1](https://huggingface.co/stabilityai/stable-diffusion-2-1) (may give unexpected results).

Let's have a look at the images (*512X512*)

| Without Feedback            | With Feedback  (1st image)          |
|---------------------|---------------------|
| ![Image 1](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/fabric_wo_feedback.jpg) | ![Feedback Image 1](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/fabric_w_feedback.png) | 

<Tip>

Make sure to check out the Stable Diffusion [Tips](overview#tips) section to learn how to explore the tradeoff between scheduler speed and quality, and how to reuse pipeline components efficiently!

</Tip>

[[autodoc]] FabricPipeline
	- all
	- __call__

## StableDiffusionPipelineOutput

[[autodoc]] pipelines.stable_diffusion.StableDiffusionPipelineOutput
