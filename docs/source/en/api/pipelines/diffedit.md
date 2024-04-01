<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# DiffEdit

[DiffEdit: Diffusion-based semantic image editing with mask guidance](https://huggingface.co/papers/2210.11427) is by Guillaume Couairon, Jakob Verbeek, Holger Schwenk, and Matthieu Cord.

The abstract from the paper is:

*Image generation has recently seen tremendous advances, with diffusion models allowing to synthesize convincing images for a large variety of text prompts. In this article, we propose DiffEdit, a method to take advantage of text-conditioned diffusion models for the task of semantic image editing, where the goal is to edit an image based on a text query. Semantic image editing is an extension of image generation, with the additional constraint that the generated image should be as similar as possible to a given input image. Current editing methods based on diffusion models usually require to provide a mask, making the task much easier by treating it as a conditional inpainting task. In contrast, our main contribution is able to automatically generate a mask highlighting regions of the input image that need to be edited, by contrasting predictions of a diffusion model conditioned on different text prompts. Moreover, we rely on latent inference to preserve content in those regions of interest and show excellent synergies with mask-based diffusion. DiffEdit achieves state-of-the-art editing performance on ImageNet. In addition, we evaluate semantic image editing in more challenging settings, using images from the COCO dataset as well as text-based generated images.*

The original codebase can be found at [Xiang-cd/DiffEdit-stable-diffusion](https://github.com/Xiang-cd/DiffEdit-stable-diffusion), and you can try it out in this [demo](https://blog.problemsolversguild.com/technical/research/2022/11/02/DiffEdit-Implementation.html).

This pipeline was contributed by [clarencechen](https://github.com/clarencechen). ❤️

## Tips

* The pipeline can generate masks that can be fed into other inpainting pipelines.
* In order to generate an image using this pipeline, both an image mask (source and target prompts can be manually specified or generated, and passed to [`~StableDiffusionDiffEditPipeline.generate_mask`])
and a set of partially inverted latents (generated using [`~StableDiffusionDiffEditPipeline.invert`]) _must_ be provided as arguments when calling the pipeline to generate the final edited image.
* The function [`~StableDiffusionDiffEditPipeline.generate_mask`] exposes two prompt arguments, `source_prompt` and `target_prompt`
that let you control the locations of the semantic edits in the final image to be generated. Let's say,
you wanted to translate from "cat" to "dog". In this case, the edit direction will be "cat -> dog". To reflect
this in the generated mask, you simply have to set the embeddings related to the phrases including "cat" to
`source_prompt` and "dog" to `target_prompt`.
* When generating partially inverted latents using `invert`, assign a caption or text embedding describing the
overall image to the `prompt` argument to help guide the inverse latent sampling process. In most cases, the
source concept is sufficiently descriptive to yield good results, but feel free to explore alternatives.
* When calling the pipeline to generate the final edited image, assign the source concept to `negative_prompt`
and the target concept to `prompt`. Taking the above example, you simply have to set the embeddings related to
the phrases including "cat" to `negative_prompt` and "dog" to `prompt`.
* If you wanted to reverse the direction in the example above, i.e., "dog -> cat", then it's recommended to:
    * Swap the `source_prompt` and `target_prompt` in the arguments to `generate_mask`.
    * Change the input prompt in [`~StableDiffusionDiffEditPipeline.invert`] to include "dog".
    * Swap the `prompt` and `negative_prompt` in the arguments to call the pipeline to generate the final edited image.
* The source and target prompts, or their corresponding embeddings, can also be automatically generated. Please refer to the [DiffEdit](../../using-diffusers/diffedit) guide for more details.

## StableDiffusionDiffEditPipeline
[[autodoc]] StableDiffusionDiffEditPipeline
    - all
    - generate_mask
    - invert
    - __call__

## StableDiffusionPipelineOutput
[[autodoc]] pipelines.stable_diffusion.StableDiffusionPipelineOutput
