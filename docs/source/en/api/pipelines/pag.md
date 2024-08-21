<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Perturbed-Attention Guidance

[Perturbed-Attention Guidance (PAG)](https://ku-cvlab.github.io/Perturbed-Attention-Guidance/) is a new diffusion sampling guidance that improves sample quality across both unconditional and conditional settings, achieving this without requiring further training or the integration of external modules.

PAG was introduced in [Self-Rectifying Diffusion Sampling with Perturbed-Attention Guidance](https://huggingface.co/papers/2403.17377) by Donghoon Ahn, Hyoungwon Cho, Jaewon Min, Wooseok Jang, Jungwoo Kim, SeonHwa Kim, Hyun Hee Park, Kyong Hwan Jin and Seungryong Kim.

The abstract from the paper is:

*Recent studies have demonstrated that diffusion models are capable of generating high-quality samples, but their quality heavily depends on sampling guidance techniques, such as classifier guidance (CG) and classifier-free guidance (CFG). These techniques are often not applicable in unconditional generation or in various downstream tasks such as image restoration. In this paper, we propose a novel sampling guidance, called Perturbed-Attention Guidance (PAG), which improves diffusion sample quality across both unconditional and conditional settings, achieving this without requiring additional training or the integration of external modules. PAG is designed to progressively enhance the structure of samples throughout the denoising process. It involves generating intermediate samples with degraded structure by substituting selected self-attention maps in diffusion U-Net with an identity matrix, by considering the self-attention mechanisms' ability to capture structural information, and guiding the denoising process away from these degraded samples. In both ADM and Stable Diffusion, PAG surprisingly improves sample quality in conditional and even unconditional scenarios. Moreover, PAG significantly improves the baseline performance in various downstream tasks where existing guidances such as CG or CFG cannot be fully utilized, including ControlNet with empty prompts and image restoration such as inpainting and deblurring.*

PAG can be used by specifying the `pag_applied_layers` as a parameter when instantiating a PAG pipeline. It can be a single string or a list of strings. Each string can be a unique layer identifier or a regular expression to identify one or more layers.

- Full identifier as a normal string: `down_blocks.2.attentions.0.transformer_blocks.0.attn1.processor`
- Full identifier as a RegEx: `down_blocks.2.(attentions|motion_modules).0.transformer_blocks.0.attn1.processor`
- Partial identifier as a RegEx: `down_blocks.2`, or `attn1`
- List of identifiers (can be combo of strings and ReGex): `["blocks.1", "blocks.(14|20)", r"down_blocks\.(2,3)"]`

<Tip warning={true}>

Since RegEx is supported as a way for matching layer identifiers, it is crucial to use it correctly otherwise there might be unexpected behaviour. The recommended way to use PAG is by specifying layers as `blocks.{layer_index}` and `blocks.({layer_index_1|layer_index_2|...})`. Using it in any other way, while doable, may bypass our basic validation checks and give you unexpected results.

</Tip>

## AnimateDiffPAGPipeline
[[autodoc]] AnimateDiffPAGPipeline
  - all
  - __call__

## HunyuanDiTPAGPipeline
[[autodoc]] HunyuanDiTPAGPipeline
  - all
  - __call__

## KolorsPAGPipeline
[[autodoc]] KolorsPAGPipeline
  - all
  - __call__

## StableDiffusionPAGPipeline
[[autodoc]] StableDiffusionPAGPipeline
	- all
	- __call__

## StableDiffusionControlNetPAGPipeline
[[autodoc]] StableDiffusionControlNetPAGPipeline
	- all
	- __call__

## StableDiffusionXLPAGPipeline
[[autodoc]] StableDiffusionXLPAGPipeline
	- all
	- __call__

## StableDiffusionXLPAGImg2ImgPipeline
[[autodoc]] StableDiffusionXLPAGImg2ImgPipeline
	- all
	- __call__

## StableDiffusionXLPAGInpaintPipeline
[[autodoc]] StableDiffusionXLPAGInpaintPipeline
	- all
	- __call__

## StableDiffusionXLControlNetPAGPipeline
[[autodoc]] StableDiffusionXLControlNetPAGPipeline
	- all
	- __call__

## StableDiffusionXLControlNetPAGImg2ImgPipeline
[[autodoc]] StableDiffusionXLControlNetPAGImg2ImgPipeline
	- all
	- __call__

## StableDiffusion3PAGPipeline
[[autodoc]] StableDiffusion3PAGPipeline
	- all
	- __call__


## PixArtSigmaPAGPipeline
[[autodoc]] PixArtSigmaPAGPipeline
	- all
	- __call__
