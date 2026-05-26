<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Philosophy

🧨 Diffusers provides **state-of-the-art** pretrained diffusion models across multiple modalities.
Its purpose is to serve as a **modular toolbox** for both inference and training.

We aim at building a library that stands the test of time and therefore take API design very seriously.

In a nutshell, Diffusers is built to be a natural extension of PyTorch. Therefore, most of our design choices are based on [PyTorch's Design Principles](https://pytorch.org/docs/stable/community/design.html#pytorch-design-philosophy). Let's go over the most important ones:

## Usability over Performance

- While Diffusers has many built-in performance-enhancing features (see [Memory and Speed](https://huggingface.co/docs/diffusers/optimization/fp16)), models are always loaded with the highest precision and lowest optimization. Therefore, by default diffusion pipelines are always instantiated on CPU with float32 precision if not otherwise defined by the user. This ensures usability across different platforms and accelerators and means that no complex installations are required to run the library.
- Diffusers aims to be a **light-weight** package and therefore has very few required dependencies, but many soft dependencies that can improve performance (such as `accelerate`, `safetensors`, `onnx`, etc...). We strive to keep the library as lightweight as possible so that it can be added without much concern as a dependency on other packages.
- Diffusers prefers simple, self-explainable code over condensed, magic code. This means that short-hand code syntaxes such as lambda functions, and advanced PyTorch operators are often not desired.

## Simple over easy

As PyTorch states, **explicit is better than implicit** and **simple is better than complex**. This design philosophy is reflected in multiple parts of the library:
- We follow PyTorch's API with methods like [`DiffusionPipeline.to`](https://huggingface.co/docs/diffusers/main/en/api/diffusion_pipeline#diffusers.DiffusionPipeline.to) to let the user handle device management.
- Raising concise error messages is preferred to silently correct erroneous input. Diffusers aims at teaching the user, rather than making the library as easy to use as possible.
- Complex model vs. scheduler logic is exposed instead of magically handled inside. Schedulers/Samplers are separated from diffusion models with minimal dependencies on each other. This forces the user to write the unrolled denoising loop. However, the separation allows for easier debugging and gives the user more control over adapting the denoising process or switching out diffusion models or schedulers.
- Separately trained components of the diffusion pipeline, *e.g.* the text encoder, the unet, and the variational autoencoder, each have their own model class. This forces the user to handle the interaction between the different model components, and the serialization format separates the model components into different files. However, this allows for easier debugging and customization. DreamBooth or Textual Inversion training
is very simple thanks to Diffusers' ability to separate single components of the diffusion pipeline.

## Tweakable, contributor-friendly over abstraction

For large parts of the library, Diffusers adopts an important design principle of the [Transformers library](https://github.com/huggingface/transformers), which is to prefer copy-pasted code over hasty abstractions. This design principle is very opinionated and stands in stark contrast to popular design principles such as [Don't repeat yourself (DRY)](https://en.wikipedia.org/wiki/Don%27t_repeat_yourself).
In short, just like Transformers does for modeling files, Diffusers prefers to keep an extremely low level of abstraction and very self-contained code for pipelines and schedulers.
Functions, long code blocks, and even classes can be copied across multiple files which at first can look like a bad, sloppy design choice that makes the library unmaintainable.
**However**, this design has proven to be extremely successful for Transformers and makes a lot of sense for community-driven, open-source machine learning libraries because:
- Machine Learning is an extremely fast-moving field in which paradigms, model architectures, and algorithms are changing rapidly, which therefore makes it very difficult to define long-lasting code abstractions.
- Machine Learning practitioners like to be able to quickly tweak existing code for ideation and research and therefore prefer self-contained code over one that contains many abstractions.
- Open-source libraries rely on community contributions and therefore must build a library that is easy to contribute to. The more abstract the code, the more dependencies, the harder to read, and the harder to contribute to. Contributors simply stop contributing to very abstract libraries out of fear of breaking vital functionality. If contributing to a library cannot break other fundamental code, not only is it more inviting for potential new contributors, but it is also easier to review and contribute to multiple parts in parallel.

At Hugging Face, we call this design the **single-file policy** which means that almost all of the code of a certain class should be written in a single, self-contained file. To read more about the philosophy, you can have a look
at [this blog post](https://huggingface.co/blog/transformers-design-philosophy).

In Diffusers, we follow this philosophy for pipelines, schedulers, and models alike. Some older models predate this convention and are kept as-is; all new model architectures live in their own self-contained files. See the [Models](#models) section below for details.

Great, now you should have generally understood why 🧨 Diffusers is designed the way it is 🤗.
We try to apply these design principles consistently across the library. Nevertheless, there are some minor exceptions to the philosophy or some unlucky design choices. If you have feedback regarding the design, we would ❤️  to hear it [directly on GitHub](https://github.com/huggingface/diffusers/issues/new?assignees=&labels=&template=feedback.md&title=).

## Design Philosophy in Details

Now, let's look a bit into the nitty-gritty details of the design philosophy. Diffusers consists of [models](https://github.com/huggingface/diffusers/tree/main/src/diffusers/models), [schedulers](https://github.com/huggingface/diffusers/tree/main/src/diffusers/schedulers), and two ways to compose them into a runnable workflow: standard [pipelines](https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines) (monolithic, one task per pipeline class) and [Modular Diffusers](../modular_diffusers/overview) (composable, block-based). Let's walk through more in-detail design decisions for each.

### Pipelines

Pipelines are intended **only** for inference. Pipelines are designed to be easy to use (therefore do not follow [*Simple over easy*](#simple-over-easy) 100%) — they should be readable, self-explanatory, easy to tweak, and loosely seen as examples of how to use [models](#models) and [schedulers](#schedulers). Pipelines are not feature complete, to build feature-complete user interfaces on top of Diffusers, consider using [Modular Diffusers](../modular_diffusers/overview).

The following design principles are followed:
- Pipelines follow the single-file policy. All pipelines can be found in individual directories under src/diffusers/pipelines. One pipeline folder corresponds to one diffusion paper/project/release. Multiple pipeline files can be gathered in one pipeline folder, as it’s done for [`src/diffusers/pipelines/stable-diffusion`](https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines/stable_diffusion). If pipelines share similar functionality, one can make use of the [# Copied from mechanism](https://github.com/huggingface/diffusers/blob/125d783076e5bd9785beb05367a2d2566843a271/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_img2img.py#L251).
- Pipelines all inherit from [`DiffusionPipeline`].
- Every pipeline consists of different model and scheduler components, that are documented in the [`model_index.json` file](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/blob/main/model_index.json), are accessible under the same name as attributes of the pipeline and can be shared between pipelines with [`DiffusionPipeline.components`](https://huggingface.co/docs/diffusers/main/en/api/diffusion_pipeline#diffusers.DiffusionPipeline.components) function.
- Every pipeline should be loadable via the [`DiffusionPipeline.from_pretrained`](https://huggingface.co/docs/diffusers/main/en/api/diffusion_pipeline#diffusers.DiffusionPipeline.from_pretrained) function.
- Every pipeline should have one and only one way to run it via a `__call__` method. The naming of the `__call__` arguments should be consistent across all pipelines.
- Pipelines should be named after the task they are intended to solve.

### Modular Pipelines

Modular Pipelines is the composable alternative to standard pipelines: workflows are built from reusable **pipeline blocks** that can be mixed, matched, swapped, and shared. Where standard pipelines are loose reference examples of how to use models and schedulers, Modular Pipelines is the recommended path for building feature-complete user interfaces on top of Diffusers, and for the community to build and share new pipelines in a decentralized way.

The following design principles are followed:
- Modular pipelines also follow the single-file policy. Each modular pipeline lives in its own folder under [`src/diffusers/modular_pipelines/`](https://github.com/huggingface/diffusers/tree/main/src/diffusers/modular_pipelines). A modular pipeline folder contains multiple files corresponding to each stage of the workflow — `encoders.py`, `before_denoise.py`, `denoise.py`, `decoders.py`, `modular_blocks_<model>.py` for assembly, and `modular_pipeline.py` for the per-model `ModularPipeline` subclass. Modular pipelines don't cross-import each other.
- Each modular pipeline is defined as a set of `ModularPipelineBlocks` — leaf blocks in the stage files (`encoders.py`, `before_denoise.py`, `denoise.py`, `decoders.py`), then assembled into the full workflow in `modular_blocks_<model>.py` using container classes like `SequentialPipelineBlocks` and `AutoPipelineBlocks`. Unlike `DiffusionPipeline`, which both defines the computation logic and runs it, modular splits these into two concepts: blocks are pure definitions (declare inputs, outputs, and component dependencies — they do not hold weights and cannot be executed), and a `ModularPipeline` is the runnable counterpart, created via `.init_pipeline(repo_id)`. Keeping blocks stateless and weight-free is what makes them freely composable, swappable, and shareable across workflows.
- To support a new workflow/task, write the task-specific block(s), compose them with existing ones, and declare the new workflow in `_workflow_map` on the top-level block assembly. Unlike `DiffusionPipeline`, each `ModularPipeline` can support many workflows (text2img, img2img, inpaint, etc.).

See the [Modular Diffusers documentation](../modular_diffusers/overview) for the full design and usage guide.

### Models

Models are designed as configurable toolboxes that are natural extensions of [PyTorch's Module class](https://pytorch.org/docs/stable/generated/torch.nn.Module.html). They should follow the **single-file policy**. Some older models predate this convention and are kept as-is as legacy exceptions — not patterns to follow for new models. E.g. the original [`UNet2DConditionModel`] class was used for different UNet variations.

The following design principles are followed:
- Each model architecture type lives in its own folder under [`src/diffusers/models`](https://github.com/huggingface/diffusers/tree/main/src/diffusers/models) (e.g. [`transformers/`](https://github.com/huggingface/diffusers/tree/main/src/diffusers/models/transformers), [`autoencoders/`](https://github.com/huggingface/diffusers/tree/main/src/diffusers/models/autoencoders), [`unets/`](https://github.com/huggingface/diffusers/tree/main/src/diffusers/models/unets)), and each model family has its own file within that folder (e.g. [`transformer_flux.py`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_flux.py), [`transformer_wan.py`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_wan.py)).
- Models follow the **single-file policy** — each model file should be self-contained, except for a small number of standard modules that every model uses identically (e.g. timestep embeddings, normalization layers), which can be imported from [`embeddings.py`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/embeddings.py) and [`normalization.py`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/normalization.py).
- Models intend to expose complexity, just like PyTorch's `Module` class, and give clear error messages.
- Models all inherit from `ModelMixin` and `ConfigMixin`.
- Models can be optimized for performance when it doesn’t demand major code changes, keeps backward compatibility, and gives significant memory or compute gain.
- Models should by default have the highest precision and lowest performance setting.
- To integrate a new model architecture that's similar to an existing one, copy the existing file as a starting point and adapt it. Use [`# Copied from`](./contribution#copied-from-mechanism) annotations on layers that remain identical so `make fix-copies` keeps them in sync.

### Schedulers

Schedulers are responsible to guide the denoising process for inference as well as to define a noise schedule for training. They are designed as individual classes with loadable configuration files and strongly follow the **single-file policy**.

The following design principles are followed:
- All schedulers are found in [`src/diffusers/schedulers`](https://github.com/huggingface/diffusers/tree/main/src/diffusers/schedulers).
- Schedulers are **not** allowed to import from large utils files and shall be kept very self-contained.
- One scheduler Python file corresponds to one scheduler algorithm (as might be defined in a paper).
- If schedulers share similar functionalities, we can make use of the `# Copied from` mechanism.
- Schedulers all inherit from `SchedulerMixin` and `ConfigMixin`.
- Schedulers can be easily swapped out with the [`ConfigMixin.from_config`](https://huggingface.co/docs/diffusers/main/en/api/configuration#diffusers.ConfigMixin.from_config) method as explained in detail [here](../using-diffusers/schedulers).
- Every scheduler has to have a `set_num_inference_steps`, and a `step` function. `set_num_inference_steps(...)` has to be called before every denoising process, *i.e.* before `step(...)` is called.
- Every scheduler exposes the timesteps to be "looped over" via a `timesteps` attribute, which is an array of timesteps the model will be called upon.
- The `step(...)` function takes a predicted model output and the "current" sample (x_t) and returns the "previous", slightly more denoised sample (x_t-1).
- Given the complexity of diffusion schedulers, the `step` function does not expose all the complexity and can be a bit of a "black box".
- In almost all cases, novel schedulers shall be implemented in a new scheduling file.