<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Quickstart

Modular Diffusers is a framework for quickly building flexible and customizable pipelines. At the core of Modular Diffusers are [`ModularPipelineBlocks`] that can be combined with other blocks to adapt to new workflows. The blocks are converted into a [`ModularPipeline`], a friendly user-facing interface developers can use.

This doc will show you how to implement a [Differential Diffusion](https://differential-diffusion.github.io/) pipeline with the modular framework.

## ModularPipelineBlocks

[`ModularPipelineBlocks`] are *definitions* that specify the components, inputs, outputs, and computation logic for a single step in a pipeline. There are four types of blocks.

- [`ModularPipelineBlocks`] is the most basic block for a single step.
- [`SequentialPipelineBlocks`] is a multi-block that composes other blocks linearly. The outputs of one block are the inputs to the next block.
- [`LoopSequentialPipelineBlocks`] is a multi-block that runs iteratively and is designed for iterative workflows.
- [`AutoPipelineBlocks`] is a collection of blocks for different workflows and it selects which block to run based on the input. It is designed to conveniently package multiple workflows into a single pipeline.

[Differential Diffusion](https://differential-diffusion.github.io/) is an image-to-image workflow. Start with the `IMAGE2IMAGE_BLOCKS` preset, a collection of `ModularPipelineBlocks` for image-to-image generation.

```py
from diffusers.modular_pipelines.stable_diffusion_xl import IMAGE2IMAGE_BLOCKS
IMAGE2IMAGE_BLOCKS = InsertableDict([
    ("text_encoder", StableDiffusionXLTextEncoderStep),
    ("image_encoder", StableDiffusionXLVaeEncoderStep),
    ("input", StableDiffusionXLInputStep),
    ("set_timesteps", StableDiffusionXLImg2ImgSetTimestepsStep),
    ("prepare_latents", StableDiffusionXLImg2ImgPrepareLatentsStep),
    ("prepare_add_cond", StableDiffusionXLImg2ImgPrepareAdditionalConditioningStep),
    ("denoise", StableDiffusionXLDenoiseStep),
    ("decode", StableDiffusionXLDecodeStep)
])
```

## Pipeline and block states

Modular Diffusers uses *state* to communicate data between blocks. There are two types of states.

- [`PipelineState`] is a global state that can be used to track all inputs and outputs across all blocks.
- [`BlockState`] is a local view of relevant variables from [`PipelineState`] for an individual block.

## Customizing blocks

[Differential Diffusion](https://differential-diffusion.github.io/) differs from standard image-to-image in its `prepare_latents` and `denoise` blocks. All the other blocks can be reused, but you'll need to modify these two.

Create placeholder `ModularPipelineBlocks` for `prepare_latents` and `denoise` by copying and modifying the existing ones.

Print the `denoise` block to see that it is composed of [`LoopSequentialPipelineBlocks`] with three sub-blocks, `before_denoiser`, `denoiser`, and `after_denoiser`. Only the `before_denoiser` sub-block needs to be modified to prepare the latent input for the denoiser based on the change map.

```py
denoise_blocks = IMAGE2IMAGE_BLOCKS["denoise"]()
print(denoise_blocks)
```

Replace the `StableDiffusionXLLoopBeforeDenoiser` sub-block with the new `SDXLDiffDiffLoopBeforeDenoiser` block.

```py
# Copy existing blocks as placeholders
class SDXLDiffDiffPrepareLatentsStep(ModularPipelineBlocks):
    """Copied from StableDiffusionXLImg2ImgPrepareLatentsStep - will modify later"""
    # ... same implementation as StableDiffusionXLImg2ImgPrepareLatentsStep

class SDXLDiffDiffDenoiseStep(StableDiffusionXLDenoiseLoopWrapper):
    block_classes = [SDXLDiffDiffLoopBeforeDenoiser, StableDiffusionXLLoopDenoiser, StableDiffusionXLLoopAfterDenoiser]
    block_names = ["before_denoiser", "denoiser", "after_denoiser"]
```

### prepare_latents

The `prepare_latents` block requires the following changes.

- a processor to process the change map
- a new `inputs` to accept the user-provided change map, `timestep` for precomputing all the latents and `num_inference_steps` to create the mask for updating the image regions
- update the computation in the `__call__` method for processing the change map and creating the masks, and storing it in the [`BlockState`]

```diff
class SDXLDiffDiffPrepareLatentsStep(ModularPipelineBlocks):
    @property
    def expected_components(self) -> List[ComponentSpec]:
        return [
            ComponentSpec("vae", AutoencoderKL),
            ComponentSpec("scheduler", EulerDiscreteScheduler),
+           ComponentSpec("mask_processor", VaeImageProcessor, config=FrozenDict({"do_normalize": False, "do_convert_grayscale": True}))
        ]
    @property
    def inputs(self) -> List[Tuple[str, Any]]:
        return [
            InputParam("generator"),
+           InputParam("diffdiff_map", required=True),
-           InputParam("latent_timestep", required=True, type_hint=torch.Tensor),
+           InputParam("timesteps", type_hint=torch.Tensor),
+           InputParam("num_inference_steps", type_hint=int),
        ]

    @property
    def intermediate_outputs(self) -> List[OutputParam]:
        return [
+           OutputParam("original_latents", type_hint=torch.Tensor),
+           OutputParam("diffdiff_masks", type_hint=torch.Tensor),
        ]
    def __call__(self, components, state: PipelineState):
        # ... existing logic ...
+       # Process change map and create masks
+       diffdiff_map = components.mask_processor.preprocess(block_state.diffdiff_map, height=latent_height, width=latent_width)
+       thresholds = torch.arange(block_state.num_inference_steps, dtype=diffdiff_map.dtype) / block_state.num_inference_steps
+       block_state.diffdiff_masks = diffdiff_map > (thresholds + (block_state.denoising_start or 0))
+       block_state.original_latents = block_state.latents
```

### denoise

The `before_denoiser` sub-block requires the following changes.

- a new `inputs` to accept a `denoising_start` parameter,  `original_latents` and `diffdiff_masks` from the `prepare_latents` block
- update the computation in the `__call__` method for applying Differential Diffusion

```diff
class SDXLDiffDiffLoopBeforeDenoiser(ModularPipelineBlocks):
    @property
    def description(self) -> str:
        return (
            "Step within the denoising loop for differential diffusion that prepare the latent input for the denoiser"
        )

    @property
    def inputs(self) -> List[str]:
        return [
            InputParam("latents", required=True, type_hint=torch.Tensor),
+           InputParam("denoising_start"),
+           InputParam("original_latents", type_hint=torch.Tensor),
+           InputParam("diffdiff_masks", type_hint=torch.Tensor),
        ]

    def __call__(self, components, block_state, i, t):
+       # Apply differential diffusion logic
+       if i == 0 and block_state.denoising_start is None:
+           block_state.latents = block_state.original_latents[:1]
+       else:
+           block_state.mask = block_state.diffdiff_masks[i].unsqueeze(0).unsqueeze(1)
+           block_state.latents = block_state.original_latents[i] * block_state.mask + block_state.latents * (1 - block_state.mask)

        # ... rest of existing logic ...
```

## Assembling the blocks

You should have all the blocks you need at this point to create a [`ModularPipeline`].

Copy the existing `IMAGE2IMAGE_BLOCKS` preset and for the `set_timesteps` block, use the `set_timesteps` from the `TEXT2IMAGE_BLOCKS` because Differential Diffusion doesn't require a `strength` parameter.

Set the `prepare_latents` and `denoise` blocks to the `SDXLDiffDiffPrepareLatentsStep` and `SDXLDiffDiffDenoiseStep` blocks you just modified.

Call [`SequentialPipelineBlocks.from_blocks_dict`] on the blocks to create a `SequentialPipelineBlocks`.

```py
DIFFDIFF_BLOCKS = IMAGE2IMAGE_BLOCKS.copy()
DIFFDIFF_BLOCKS["set_timesteps"] = TEXT2IMAGE_BLOCKS["set_timesteps"]
DIFFDIFF_BLOCKS["prepare_latents"] = SDXLDiffDiffPrepareLatentsStep
DIFFDIFF_BLOCKS["denoise"] = SDXLDiffDiffDenoiseStep

dd_blocks = SequentialPipelineBlocks.from_blocks_dict(DIFFDIFF_BLOCKS)
print(dd_blocks)
```

## ModularPipeline

Convert the [`SequentialPipelineBlocks`] into a [`ModularPipeline`] with the [`ModularPipeline.init_pipeline`] method. This initializes the expected components to load from a `modular_model_index.json` file. Explicitly load the components by calling [`ModularPipeline.load_components`].

It is a good idea to initialize the [`ComponentManager`] with the pipeline to help manage the different components. Once you call [`~ModularPipeline.load_components`], the components are registered to the [`ComponentManager`] and can be shared between workflows. The example below uses the `collection` argument to assign the components a `"diffdiff"` label for better organization.

```py
from diffusers.modular_pipelines import ComponentsManager

components = ComponentManager()

dd_pipeline = dd_blocks.init_pipeline("YiYiXu/modular-demo-auto", components_manager=components, collection="diffdiff")
dd_pipeline.load_default_componenets(torch_dtype=torch.float16)
dd_pipeline.to("cuda")
```

## Adding workflows

Other workflows can be added to the [`ModularPipeline`] to support additional features without rewriting the entire pipeline from scratch.

This section demonstrates how to add an IP-Adapter or ControlNet.

### IP-Adapter

Stable Diffusion XL already has a preset IP-Adapter block that you can use and doesn't require any changes to the existing Differential Diffusion pipeline.

```py
from diffusers.modular_pipelines.stable_diffusion_xl.encoders import StableDiffusionXLAutoIPAdapterStep

ip_adapter_block = StableDiffusionXLAutoIPAdapterStep()
```

Use the [`sub_blocks.insert`] method to insert it into the [`ModularPipeline`]. The example below inserts the `ip_adapter_block` at position `0`. Print the pipeline to see that the `ip_adapter_block` is added and it requires an `ip_adapter_image`. This also added two components to the pipeline, the `image_encoder` and `feature_extractor`.

```py
dd_blocks.sub_blocks.insert("ip_adapter", ip_adapter_block, 0)
```

Call [`~ModularPipeline.init_pipeline`] to initialize a [`ModularPipeline`] and use [`~ModularPipeline.load_components`] to load the model components. Load and set the IP-Adapter to run the pipeline.

```py
dd_pipeline = dd_blocks.init_pipeline("YiYiXu/modular-demo-auto", collection="diffdiff")
dd_pipeline.load_components(torch_dtype=torch.float16)
dd_pipeline.loader.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")
dd_pipeline.loader.set_ip_adapter_scale(0.6)
dd_pipeline = dd_pipeline.to(device)

ip_adapter_image = load_image("https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/diffdiff_orange.jpeg")
image = load_image("https://huggingface.co/datasets/OzzyGT/testing-resources/resolve/main/differential/20240329211129_4024911930.png?download=true")
mask = load_image("https://huggingface.co/datasets/OzzyGT/testing-resources/resolve/main/differential/gradient_mask.png?download=true")

prompt = "a green pear"
negative_prompt = "blurry"
generator = torch.Generator(device=device).manual_seed(42)

image = dd_pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=25,
    generator=generator,
    ip_adapter_image=ip_adapter_image,
    diffdiff_map=mask,
    image=image,
    output="images"
)[0]
```

### ControlNet

Stable Diffusion XL already has a preset ControlNet block that can readily be used.

```py
from diffusers.modular_pipelines.stable_diffusion_xl.modular_blocks import StableDiffusionXLAutoControlNetInputStep

control_input_block = StableDiffusionXLAutoControlNetInputStep()
```

However, it requires modifying the `denoise` block because that's where the ControlNet injects the control information into the UNet.

Modify the `denoise` block by replacing the `StableDiffusionXLLoopDenoiser` sub-block with the `StableDiffusionXLControlNetLoopDenoiser`.

```py
class SDXLDiffDiffControlNetDenoiseStep(StableDiffusionXLDenoiseLoopWrapper):
    block_classes = [SDXLDiffDiffLoopBeforeDenoiser, StableDiffusionXLControlNetLoopDenoiser, StableDiffusionXLDenoiseLoopAfterDenoiser]
    block_names = ["before_denoiser", "denoiser", "after_denoiser"]

controlnet_denoise_block = SDXLDiffDiffControlNetDenoiseStep()
```

Insert the `controlnet_input` block and replace the `denoise` block with the new `controlnet_denoise_block`. Initialize a [`ModularPipeline`] and [`~ModularPipeline.load_components`] into it.

```py
dd_blocks.sub_blocks.insert("controlnet_input", control_input_block, 7)
dd_blocks.sub_blocks["denoise"] = controlnet_denoise_block

dd_pipeline = dd_blocks.init_pipeline("YiYiXu/modular-demo-auto", collection="diffdiff")
dd_pipeline.load_components(torch_dtype=torch.float16)
dd_pipeline = dd_pipeline.to(device)

control_image = load_image("https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/diffdiff_tomato_canny.jpeg")
image = load_image("https://huggingface.co/datasets/OzzyGT/testing-resources/resolve/main/differential/20240329211129_4024911930.png?download=true")
mask = load_image("https://huggingface.co/datasets/OzzyGT/testing-resources/resolve/main/differential/gradient_mask.png?download=true")

prompt = "a green pear"
negative_prompt = "blurry"
generator = torch.Generator(device=device).manual_seed(42)

image = dd_pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=25,
    generator=generator,
    control_image=control_image,
    controlnet_conditioning_scale=0.5,
    diffdiff_map=mask,
    image=image,
    output="images"
)[0]
```

### AutoPipelineBlocks

The Differential Diffusion, IP-Adapter, and ControlNet workflows can be bundled into a single [`ModularPipeline`] by using [`AutoPipelineBlocks`]. This allows automatically selecting which sub-blocks to run based on the inputs like `control_image` or `ip_adapter_image`. If none of these inputs are passed, then it defaults to the Differential Diffusion.

Use `block_trigger_inputs` to only run the `SDXLDiffDiffControlNetDenoiseStep` block if a `control_image` input is provided. Otherwise, the `SDXLDiffDiffDenoiseStep` is used.

```py
class SDXLDiffDiffAutoDenoiseStep(AutoPipelineBlocks):
    block_classes = [SDXLDiffDiffControlNetDenoiseStep, SDXLDiffDiffDenoiseStep]
    block_names = ["controlnet_denoise", "denoise"]
    block_trigger_inputs = ["controlnet_cond", None]
```

Add the `ip_adapter` and `controlnet_input` blocks.

```py
DIFFDIFF_AUTO_BLOCKS = IMAGE2IMAGE_BLOCKS.copy()
DIFFDIFF_AUTO_BLOCKS["prepare_latents"] = SDXLDiffDiffPrepareLatentsStep
DIFFDIFF_AUTO_BLOCKS["set_timesteps"] = TEXT2IMAGE_BLOCKS["set_timesteps"]
DIFFDIFF_AUTO_BLOCKS["denoise"] = SDXLDiffDiffAutoDenoiseStep
DIFFDIFF_AUTO_BLOCKS.insert("ip_adapter", StableDiffusionXLAutoIPAdapterStep, 0)
DIFFDIFF_AUTO_BLOCKS.insert("controlnet_input",StableDiffusionXLControlNetAutoInput, 7)
```

Call [`SequentialPipelineBlocks.from_blocks_dict`] to create a [`SequentialPipelineBlocks`] and create a [`ModularPipeline`] and load in the model components to run.

```py
dd_auto_blocks = SequentialPipelineBlocks.from_blocks_dict(DIFFDIFF_AUTO_BLOCKS)
dd_pipeline = dd_auto_blocks.init_pipeline("YiYiXu/modular-demo-auto", collection="diffdiff")
dd_pipeline.load_components(torch_dtype=torch.float16)
```

## Share

Add your [`ModularPipeline`] to the Hub with [`~ModularPipeline.save_pretrained`] and set `push_to_hub` argument to `True`.

```py
dd_pipeline.save_pretrained("YiYiXu/test_modular_doc", push_to_hub=True)
```

Other users can load the [`ModularPipeline`] with [`~ModularPipeline.from_pretrained`].

```py
import torch
from diffusers.modular_pipelines import ModularPipeline, ComponentsManager

components = ComponentsManager()

diffdiff_pipeline = ModularPipeline.from_pretrained("YiYiXu/modular-diffdiff-0704", trust_remote_code=True, components_manager=components, collection="diffdiff")
diffdiff_pipeline.load_components(torch_dtype=torch.float16)
```
