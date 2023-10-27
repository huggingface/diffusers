<!--Copyright 2023 ParaDiGMS authors and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Parallel Sampling of Diffusion Models

[Parallel Sampling of Diffusion Models](https://huggingface.co/papers/2305.16317) is by Andy Shih, Suneel Belkhale, Stefano Ermon, Dorsa Sadigh, Nima Anari.

The abstract from the paper is:

*Diffusion models are powerful generative models but suffer from slow sampling, often taking 1000 sequential denoising steps for one sample. As a result, considerable efforts have been directed toward reducing the number of denoising steps, but these methods hurt sample quality. Instead of reducing the number of denoising steps (trading quality for speed), in this paper we explore an orthogonal approach: can we run the denoising steps in parallel (trading compute for speed)? In spite of the sequential nature of the denoising steps, we show that surprisingly it is possible to parallelize sampling via Picard iterations, by guessing the solution of future denoising steps and iteratively refining until convergence. With this insight, we present ParaDiGMS, a novel method to accelerate the sampling of pretrained diffusion models by denoising multiple steps in parallel. ParaDiGMS is the first diffusion sampling method that enables trading compute for speed and is even compatible with existing fast sampling techniques such as DDIM and DPMSolver. Using ParaDiGMS, we improve sampling speed by 2-4x across a range of robotics and image generation models, giving state-of-the-art sampling speeds of 0.2s on 100-step DiffusionPolicy and 16s on 1000-step StableDiffusion-v2 with no measurable degradation of task reward, FID score, or CLIP score.*

The original codebase can be found at [AndyShih12/paradigms](https://github.com/AndyShih12/paradigms), and the pipeline was contributed by [AndyShih12](https://github.com/AndyShih12). ❤️

## Tips

This pipeline improves sampling speed by running denoising steps in parallel, at the cost of increased total FLOPs.
Therefore, it is better to call this pipeline when running on multiple GPUs. Otherwise, without enough GPU bandwidth
sampling may be even slower than sequential sampling.

The two parameters to play with are `parallel` (batch size) and `tolerance`. 
- If it fits in memory, for a 1000-step DDPM you can aim for a batch size of around 100 
(for example, 8 GPUs and `batch_per_device=12` to get `parallel=96`). A higher batch size
may not fit in memory, and lower batch size gives less parallelism. 
- For tolerance, using a higher tolerance may get better speedups but can risk sample quality degradation. 
If there is quality degradation with the default tolerance, then use a lower tolerance like `0.001`.

For a 1000-step DDPM on 8 A100 GPUs, you can expect around a 3x speedup from [`StableDiffusionParadigmsPipeline`] compared to the [`StableDiffusionPipeline`]
by setting `parallel=80` and `tolerance=0.1`.

🤗 Diffusers offers [distributed inference support](../training/distributed_inference) for generating multiple prompts
in parallel on multiple GPUs. But [`StableDiffusionParadigmsPipeline`] is designed for speeding up sampling of a single prompt by using multiple GPUs.

<Tip>

Make sure to check out the Schedulers [guide](../../using-diffusers/schedulers) to learn how to explore the tradeoff between scheduler speed and quality, and see the [reuse components across pipelines](../../using-diffusers/loading#reuse-components-across-pipelines) section to learn how to efficiently load the same components into multiple pipelines.

</Tip>

## StableDiffusionParadigmsPipeline
[[autodoc]] StableDiffusionParadigmsPipeline
	- __call__
	- all

## StableDiffusionPipelineOutput
[[autodoc]] pipelines.stable_diffusion.StableDiffusionPipelineOutput
