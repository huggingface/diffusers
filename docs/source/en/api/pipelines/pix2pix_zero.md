<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Pix2Pix Zero

[Zero-shot Image-to-Image Translation](https://huggingface.co/papers/2302.03027) is by Gaurav Parmar, Krishna Kumar Singh, Richard Zhang, Yijun Li, Jingwan Lu, and Jun-Yan Zhu.

The abstract from the paper is:

*Large-scale text-to-image generative models have shown their remarkable ability to synthesize diverse and high-quality images. However, it is still challenging to directly apply these models for editing real images for two reasons. First, it is hard for users to come up with a perfect text prompt that accurately describes every visual detail in the input image. Second, while existing models can introduce desirable changes in certain regions, they often dramatically alter the input content and introduce unexpected changes in unwanted regions. In this work, we propose pix2pix-zero, an image-to-image translation method that can preserve the content of the original image without manual prompting. We first automatically discover editing directions that reflect desired edits in the text embedding space. To preserve the general content structure after editing, we further propose cross-attention guidance, which aims to retain the cross-attention maps of the input image throughout the diffusion process. In addition, our method does not need additional training for these edits and can directly use the existing pre-trained text-to-image diffusion model. We conduct extensive experiments and show that our method outperforms existing and concurrent works for both real and synthetic image editing.*

You can find additional information about Pix2Pix Zero on the [project page](https://pix2pixzero.github.io/),  [original codebase](https://github.com/pix2pixzero/pix2pix-zero), and try it out in a [demo](https://huggingface.co/spaces/pix2pix-zero-library/pix2pix-zero-demo).

## Tips 

* The pipeline can be conditioned on real input images.
* The pipeline exposes two main arguments, `source_embeds` and `target_embeds`,
that let you control the direction of the semantic edits in the final image to be generated. Let's say,
you want to translate from "cat" to "dog". In this case, the edit direction is "cat -> dog". To reflect
this in the pipeline, set the embeddings related to phrases including "cat" to
`source_embeds` and "dog" to `target_embeds`.
* When you're using this pipeline from a prompt, specify the _source_ concept in the prompt. Taking
the above example, a valid input prompt would be "a high resolution painting of a **cat** in the style of van gogh".
* If you wanted to reverse the direction in the example above, "dog -> cat", then it's recommended to:
    * Swap the `source_embeds` and `target_embeds`.
    * Change the input prompt to include "dog".  
* To learn more about how the source and target embeddings are generated, refer to the [original 
paper](https://huggingface.co/papers/2302.03027).
* Note that the quality of the outputs generated with this pipeline is dependent on how good the `source_embeds` and `target_embeds` are. Please refer to the [Generate source and target embeddings](/using-diffusers/pix2pix-zero) guide to learn more.

<Tip>

Make sure to check out the Schedulers [guide](/using-diffusers/schedulers) to learn how to explore the tradeoff between scheduler speed and quality, and see the [reuse components across pipelines](/using-diffusers/loading#reuse-components-across-pipelines) section to learn how to efficiently load the same components into multiple pipelines.

</Tip>

## StableDiffusionPix2PixZeroPipeline
[[autodoc]] StableDiffusionPix2PixZeroPipeline
	- __call__
	- all

## StableDiffusionPipelineOutput
[[autodoc]] pipelines.stable_diffusion.StableDiffusionPipelineOutput