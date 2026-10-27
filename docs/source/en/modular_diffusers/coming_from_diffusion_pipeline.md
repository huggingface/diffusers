<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Coming from DiffusionPipeline

If you are familiar with [`DiffusionPipeline`] and are working with a [`ModularPipeline`] for the first time, a few things work differently. They are listed here so that you have them in mind before you go through the [quickstart](./quickstart).

- **Loading is lazy and flexible.** [`~ModularPipeline.from_pretrained`] builds the pipeline from the repository's `modular_model_index.json` without loading any weights. [`~ModularPipeline.load_components`] then loads them, and it doesn't have to be all at once: by default it loads everything, but you can load only the components you name with `names`, or the set one workflow needs with `workflow`. [`~ModularPipeline.update_components`] lets you reuse a component you already have, or one that differs from the repository's spec. See [Loading components](./modular_pipeline#loading-components) and [Updating components](./modular_pipeline#updating-components).
- **[`ModularPipeline`] reads a different index, `modular_model_index.json`**, but it also works with a repository that only has a `model_index.json`, as long as the model has a modular pipeline. The reverse is not true: [`DiffusionPipeline`] and the `AutoPipeline` classes don't work with a repository that only has a `modular_model_index.json`.
- **A call returns the pipeline state**, which holds the intermediate values too (`prompt_embeds`, `latents`, ...). Ask for what you need with the `output` argument, `output="images"` or `output=["images", "latents"]`, or read an attribute of the state, as in the [quickstart](./quickstart#run-a-pipeline). See [States](./modular_diffusers_states).
- **Classifier-free guidance is supported in the `guider` component**, not a `guidance_scale` call argument: `pipe.update_components(guider=ClassifierFreeGuidance(guidance_scale=5.0))`. See the [guiders guide](../using-diffusers/guiders).
- **There is no `callback_on_step_end`**; however, you can insert a block into the denoising loop to use your existing callback functions. See [Modular pipelines](../using-diffusers/callback#modular-pipelines) in the callbacks guide.
- **The pipeline is a graph of blocks** that you can change (pop, insert, replace) before building it with [`~ModularPipelineBlocks.init_pipeline`]; `pipe.blocks` returns a copy, so edit the blocks first. See [Compose new workflows](./quickstart#compose-new-workflows).
- **Offloading is handled by a [`ComponentsManager`]**, which can serve several pipelines that share components. See the [ComponentsManager guide](./components_manager#offloading).
