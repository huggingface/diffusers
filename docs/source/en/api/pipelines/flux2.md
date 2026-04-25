<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Flux2

<div class="flex flex-wrap space-x-1">
  <img alt="LoRA" src="https://img.shields.io/badge/LoRA-d8b4fe?style=flat"/>
  <img alt="MPS" src="https://img.shields.io/badge/MPS-000000?style=flat&logo=apple&logoColor=white%22">
</div>

Flux.2 is the recent series of image generation models from Black Forest Labs, preceded by the [Flux.1](./flux.md) series. It is an entirely new model with a new architecture and pre-training done from scratch!

Original model checkpoints for Flux can be found [here](https://huggingface.co/black-forest-labs). Original inference code can be found [here](https://github.com/black-forest-labs/flux2).

> [!TIP]
> Flux2 can be quite expensive to run on consumer hardware devices. However, you can perform a suite of optimizations to run it faster and in a more memory-friendly manner. Check out [this section](https://huggingface.co/blog/sd3#memory-optimizations-for-sd3) for more details. Additionally, Flux can benefit from quantization for memory efficiency with a trade-off in inference latency. Refer to [this blog post](https://huggingface.co/blog/quanto-diffusers) to learn more.
>
> [Caching](../../optimization/cache) may also speed up inference by storing and reusing intermediate outputs.

## Caption upsampling

Flux.2 can potentially generate better better outputs with better prompts. We can "upsample"
an input prompt by setting the `caption_upsample_temperature` argument in the pipeline call arguments.
The [official implementation](https://github.com/black-forest-labs/flux2/blob/5a5d316b1b42f6b59a8c9194b77c8256be848432/src/flux2/text_encoder.py#L140) recommends this value to be 0.15.

## Reference conditioning vs. img2img

The `image` argument on `Flux2Pipeline` and `Flux2KleinPipeline` is **reference conditioning**, not
img2img. Reference images are encoded into additional attention tokens that flow through the
transformer alongside the text prompt — there is no noisy latent initialization, and so no `strength`
parameter to scale.

This differs from `StableDiffusionImg2ImgPipeline`, `FluxImg2ImgPipeline`, and
`FluxKontextInpaintPipeline`, which add noise to a latent encoding of the input image and then
partially denoise it. If you port code from those pipelines and pass `strength=...` to a Flux.2
pipeline, you will see:

```
TypeError: Flux2Pipeline.__call__() got an unexpected keyword argument 'strength'
```

Drop the `strength` kwarg and pass references via `image=` (a single image, or a list for multiple
references). For Flux.2 inpainting (which does add noise to a latent and therefore does take a
`strength` parameter), use `Flux2KleinInpaintPipeline` instead.

## Flux2Pipeline

[[autodoc]] Flux2Pipeline
	- all
	- __call__

## Flux2KleinPipeline

[[autodoc]] Flux2KleinPipeline
	- all
	- __call__

## Flux2KleinKVPipeline

[[autodoc]] Flux2KleinKVPipeline
	- all
	- __call__