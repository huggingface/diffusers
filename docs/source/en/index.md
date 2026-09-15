<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

<p align="center">
    <br>
    <img src="https://raw.githubusercontent.com/huggingface/diffusers/77aadfee6a891ab9fcfb780f87c693f7a5beeb8e/docs/source/imgs/diffusers_library.jpg" width="400" style="border: none;"/>
    <br>
</p>

# Diffusers

Diffusers provides pretrained diffusion models and the building blocks for custom image, video, and audio workflows.

It has two main paths.

- [`DiffusionPipeline`] supports few-line inference with pretrained checkpoints, plus adapters like LoRA. This is the easy path for generation.
- [Modular Diffusers](./modular_diffusers/overview) enables composable blocks and [`ModularPipeline`] for custom pipelines when you need more control.

Optimizations such as offloading and quantization keep large models runnable on memory-constrained devices. If memory is not an issue, Diffusers also supports `torch.compile` for faster inference.

Browse trending Diffusers models on the [Hub](https://huggingface.co/models?library=diffusers&sort=trending) now.

## Learn

If you're a beginner, start with the [Hugging Face Diffusion Models Course](https://huggingface.co/learn/diffusion-course/unit0/1). It covers diffusion theory and how to generate images, fine-tune models, and more with Diffusers.

The [Quickstart](./quicktour) also includes a copyable agent setup prompt for inference.

## Where next

- [Inference](./using-diffusers/loading) — load pipelines and run generation
- [Optimize and scale](./stable_diffusion) — memory, speed, quantization, and serving
- [Modular Diffusers](./modular_diffusers/overview) — build custom pipelines from blocks
- [Train and fine-tune](./training/overview) — train diffusion models and adapters
