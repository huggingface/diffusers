<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Pipelines

Pipelines provide a simple way to run state-of-the-art diffusion models in inference by bundling all of the necessary components (multiple independently-trained models, schedulers, and processors) into a single end-to-end class. Pipelines are flexible and they can be adapted to use different schedulers or even model components.

All pipelines are built from the base [`DiffusionPipeline`] class which provides basic functionality for loading, downloading, and saving all the components. Specific pipeline types (for example [`StableDiffusionPipeline`]) loaded with [`~DiffusionPipeline.from_pretrained`] are automatically detected and the pipeline components are loaded and passed to the `__init__` function of the pipeline.

<Tip warning={true}>

You shouldn't use the [`DiffusionPipeline`] class for training. Individual components (for example, [`UNet2DModel`] and [`UNet2DConditionModel`]) of diffusion pipelines are usually trained individually, so we suggest directly working with them instead.

<br>

Pipelines do not offer any training functionality. You'll notice PyTorch's autograd is disabled by decorating the [`~DiffusionPipeline.__call__`] method with a [`torch.no_grad`](https://pytorch.org/docs/stable/generated/torch.no_grad.html) decorator because pipelines should not be used for training. If you're interested in training, please take a look at the [Training](../../training/overview) guides instead!

</Tip>

The table below lists all the pipelines currently available in 🤗 Diffusers and the tasks they support. Click on a pipeline to view its abstract and published paper.

| Pipeline | Tasks |
|---|---|
| [AltDiffusion](alt_diffusion) | image2image |
| [Attend-and-Excite](attend_and_excite) | text2image |
| [Audio Diffusion](audio_diffusion) | image2audio |
| [AudioLDM](audioldm) | text2audio |
| [AudioLDM2](audioldm2) | text2audio |
| [BLIP Diffusion](blip_diffusion) | text2image |
| [Consistency Models](consistency_models) | unconditional image generation |
| [ControlNet](controlnet) | text2image, image2image, inpainting |
| [ControlNet with Stable Diffusion XL](controlnet_sdxl) | text2image |
| [Cycle Diffusion](cycle_diffusion) | image2image |
| [Dance Diffusion](dance_diffusion) | unconditional audio generation |
| [DDIM](ddim) | unconditional image generation |
| [DDPM](ddpm) | unconditional image generation |
| [DeepFloyd IF](deepfloyd_if) | text2image, image2image, inpainting, super-resolution |
| [DiffEdit](diffedit) | inpainting |
| [DiT](dit) | text2image |
| [GLIGEN](gligen) | text2image |
| [InstructPix2Pix](pix2pix) | image editing |
| [Kandinsky](kandinsky) | text2image, image2image, inpainting, interpolation |
| [Kandinsky 2.2](kandinsky_v22) | text2image, image2image, inpainting |
| [Latent Diffusion](latent_diffusion) | text2image, super-resolution |
| [LDM3D](ldm3d_diffusion) | text2image, text-to-3D |
| [MultiDiffusion](panorama) | text2image |
| [MusicLDM](musicldm) | text2audio |
| [PaintByExample](paint_by_example) | inpainting |
| [ParaDiGMS](paradigms) | text2image |
| [Pix2Pix Zero](pix2pix_zero) | image editing |
| [PNDM](pndm) | unconditional image generation |
| [RePaint](repaint) | inpainting |
| [ScoreSdeVe](score_sde_ve) | unconditional image generation |
| [Self-Attention Guidance](self_attention_guidance) | text2image |
| [Semantic Guidance](semantic_stable_diffusion) | text2image |
| [Shap-E](shap_e) | text-to-3D, image-to-3D |
| [Spectrogram Diffusion](spectrogram_diffusion) |  |
| [Stable Diffusion](stable_diffusion/overview) | text2image, image2image, depth2image, inpainting, image variation, latent upscaler, super-resolution |
| [Stable Diffusion Model Editing](model_editing) | model editing |
| [Stable Diffusion XL](stable_diffusion_xl) | text2image, image2image, inpainting |
| [Stable unCLIP](stable_unclip) | text2image, image variation |
| [KarrasVe](karras_ve) | unconditional image generation |
| [T2I Adapter](adapter) | text2image |
| [Text2Video](text_to_video) | text2video, video2video |
| [Text2Video Zero](text_to_video_zero) | text2video |
| [UnCLIP](unclip) | text2image, image variation |
| [Unconditional Latent Diffusion](latent_diffusion_uncond) | unconditional image generation |
| [UniDiffuser](unidiffuser) | text2image, image2text, image variation, text variation, unconditional image generation, unconditional audio generation |
| [Value-guided planning](value_guided_sampling) | value guided sampling |
| [Versatile Diffusion](versatile_diffusion) | text2image, image variation |
| [VQ Diffusion](vq_diffusion) | text2image |
| [Wuerstchen](wuerstchen) | text2image |

## DiffusionPipeline

[[autodoc]] DiffusionPipeline
	- all
	- __call__
	- device
	- to
	- components

## FlaxDiffusionPipeline

[[autodoc]] pipelines.pipeline_flax_utils.FlaxDiffusionPipeline

## PushToHubMixin

[[autodoc]] utils.PushToHubMixin
