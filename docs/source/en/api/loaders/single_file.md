<!--Copyright 2025 The HuggingFace Team. All rights reserved.

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

Diffusers model components with a registered bidirectional conversion expose
[`~loaders.FromOriginalModelMixin.from_single_file`], including transformers, UNets, autoencoders, ControlNets,
adapters, and audio components. See the [component registry](https://github.com/huggingface/diffusers/tree/main/src/diffusers/loaders/conversion)
for the supported classes and original format variants.

Pass the matching Diffusers component configuration explicitly when loading a component whose architecture cannot
be inferred from its checkpoint. Existing Stable Diffusion and other established checkpoint detection paths remain
available.

```python
from diffusers import DiTTransformer2DModel

transformer = DiTTransformer2DModel.from_single_file(
    "./original-transformer.safetensors",
    config="./dit/transformer",
    local_files_only=True,
)
```

Component support does not add single-file loading to its full pipeline. Transformers-owned text/vision encoders and
LoRA layouts use the conversion API or their existing pipeline loaders.

## FromSingleFileMixin

[[autodoc]] loaders.single_file.FromSingleFileMixin

## FromOriginalModelMixin

[[autodoc]] loaders.single_file_model.FromOriginalModelMixin
