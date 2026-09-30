<!--Copyright 2025 The HuggingFace Team. All rights reserved.

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

> [!WARNING]
> You shouldn't use the [`DiffusionPipeline`] class for training. Individual components (for example, [`UNet2DModel`] and [`UNet2DConditionModel`]) of diffusion pipelines are usually trained individually, so we suggest directly working with them instead.
>
> <br>
>
> Pipelines do not offer any training functionality. You'll notice PyTorch's autograd is disabled by decorating the [`~DiffusionPipeline.__call__`] method with a [`torch.no_grad`](https://pytorch.org/docs/stable/generated/torch.no_grad.html) decorator because pipelines should not be used for training. If you're interested in training, please take a look at the [Training](../../training/overview) guides instead!

Use **API > Pipelines** in the sidebar to browse all pipeline references by modality. For task-oriented workflows, start with the [Quickstart](../../quicktour) or a guide such as [text-to-image](../../using-diffusers/conditional_image_generation), [image-to-image](../../using-diffusers/img2img), [inpainting](../../using-diffusers/inpaint), or [video generation](../../using-diffusers/text-img2vid).

## DiffusionPipeline

[[autodoc]] DiffusionPipeline
	- all
	- __call__
	- device
	- to
	- components


[[autodoc]] pipelines.StableDiffusionMixin.enable_freeu

[[autodoc]] pipelines.StableDiffusionMixin.disable_freeu

## PushToHubMixin

[[autodoc]] utils.PushToHubMixin

## Callbacks

[[autodoc]] callbacks.PipelineCallback

[[autodoc]] callbacks.SDCFGCutoffCallback

[[autodoc]] callbacks.SDXLCFGCutoffCallback

[[autodoc]] callbacks.SDXLControlnetCFGCutoffCallback

[[autodoc]] callbacks.IPAdapterScaleCutoffCallback

[[autodoc]] callbacks.SD3CFGCutoffCallback
