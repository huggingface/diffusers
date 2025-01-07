<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# I2VGen-XL

[I2VGen-XL: High-Quality Image-to-Video Synthesis via Cascaded Diffusion Models](https://hf.co/papers/2311.04145.pdf) by Shiwei Zhang, Jiayu Wang, Yingya Zhang, Kang Zhao, Hangjie Yuan, Zhiwu Qin, Xiang Wang, Deli Zhao, and Jingren Zhou.

The abstract from the paper is:

*Video synthesis has recently made remarkable strides benefiting from the rapid development of diffusion models. However, it still encounters challenges in terms of semantic accuracy, clarity and spatio-temporal continuity. They primarily arise from the scarcity of well-aligned text-video data and the complex inherent structure of videos, making it difficult for the model to simultaneously ensure semantic and qualitative excellence. In this report, we propose a cascaded I2VGen-XL approach that enhances model performance by decoupling these two factors and ensures the alignment of the input data by utilizing static images as a form of crucial guidance. I2VGen-XL consists of two stages: i) the base stage guarantees coherent semantics and preserves content from input images by using two hierarchical encoders, and ii) the refinement stage enhances the video's details by incorporating an additional brief text and improves the resolution to 1280×720. To improve the diversity, we collect around 35 million single-shot text-video pairs and 6 billion text-image pairs to optimize the model. By this means, I2VGen-XL can simultaneously enhance the semantic accuracy, continuity of details and clarity of generated videos. Through extensive experiments, we have investigated the underlying principles of I2VGen-XL and compared it with current top methods, which can demonstrate its effectiveness on diverse data. The source code and models will be publicly available at [this https URL](https://i2vgen-xl.github.io/).*

The original codebase can be found [here](https://github.com/ali-vilab/i2vgen-xl/). The model checkpoints can be found [here](https://huggingface.co/ali-vilab/).

<Tip>

Make sure to check out the Schedulers [guide](../../using-diffusers/schedulers) to learn how to explore the tradeoff between scheduler speed and quality, and see the [reuse components across pipelines](../../using-diffusers/loading#reuse-a-pipeline) section to learn how to efficiently load the same components into multiple pipelines. Also, to know more about reducing the memory usage of this pipeline, refer to the ["Reduce memory usage"] section [here](../../using-diffusers/svd#reduce-memory-usage).

</Tip>

Sample output with I2VGenXL:

<table>
    <tr>
        <td><center>
        library.
        <br>
        <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/i2vgen-xl-example.gif"
            alt="library"
            style="width: 300px;" />
        </center></td>
    </tr>
</table>

## Notes

* I2VGenXL always uses a `clip_skip` value of 1. This means it leverages the penultimate layer representations from the text encoder of CLIP.
* It can generate videos of quality that is often on par with [Stable Video Diffusion](../../using-diffusers/svd) (SVD).
* Unlike SVD, it additionally accepts text prompts as inputs.
* It can generate higher resolution videos.
* When using the [`DDIMScheduler`] (which is default for this pipeline), less than 50 steps for inference leads to bad results.
* This implementation is 1-stage variant of I2VGenXL. The main figure in the [I2VGen-XL](https://arxiv.org/abs/2311.04145) paper shows a 2-stage variant, however, 1-stage variant works well. See [this discussion](https://github.com/huggingface/diffusers/discussions/7952) for more details.

## I2VGenXLPipeline
[[autodoc]] I2VGenXLPipeline
	- all
	- __call__

## I2VGenXLPipelineOutput
[[autodoc]] pipelines.i2vgen_xl.pipeline_i2vgen_xl.I2VGenXLPipelineOutput