<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Single files

The [`~loaders.FromSingleFileMixin.from_single_file`] method allows you to load:

* a model stored in a single file, which is useful if you're working with models from the diffusion ecosystem, like Automatic1111, and commonly rely on a single-file layout to store and share models
* a model stored in their originally distributed layout, which is useful if you're working with models finetuned with other services, and want to load it directly into Diffusers model objects and pipelines

> [!TIP]
> Read the [Model files and layouts](../../using-diffusers/other-formats) guide to learn more about the Diffusers-multifolder layout versus the single-file layout, and how to load models stored in these different layouts.

## Supported pipelines

- [`CogVideoXPipeline`]
- [`StableDiffusionPipeline`]
- [`StableDiffusionImg2ImgPipeline`]
- [`StableDiffusionInpaintPipeline`]
- [`StableDiffusionControlNetPipeline`]
- [`StableDiffusionControlNetImg2ImgPipeline`]
- [`StableDiffusionControlNetInpaintPipeline`]
- [`StableDiffusionUpscalePipeline`]
- [`StableDiffusionXLPipeline`]
- [`StableDiffusionXLImg2ImgPipeline`]
- [`StableDiffusionXLInpaintPipeline`]
- [`StableDiffusionXLInstructPix2PixPipeline`]
- [`StableDiffusionXLControlNetPipeline`]
- [`StableDiffusionXLKDiffusionPipeline`]
- [`StableDiffusion3Pipeline`]
- [`LatentConsistencyModelPipeline`]
- [`LatentConsistencyModelImg2ImgPipeline`]
- [`StableDiffusionControlNetXSPipeline`]
- [`StableDiffusionXLControlNetXSPipeline`]
- [`LEditsPPPipelineStableDiffusion`]
- [`LEditsPPPipelineStableDiffusionXL`]
- [`PIAPipeline`]

## Supported models

- [`UNet2DConditionModel`]
- [`StableCascadeUNet`]
- [`AutoencoderKL`]
- [`AutoencoderKLCogVideoX`]
- [`ControlNetModel`]
- [`SD3Transformer2DModel`]
- [`FluxTransformer2DModel`]

## FromSingleFileMixin

[[autodoc]] loaders.single_file.FromSingleFileMixin

## FromOriginalModelMixin

[[autodoc]] loaders.single_file_model.FromOriginalModelMixin
