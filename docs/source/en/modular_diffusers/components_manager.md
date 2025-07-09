<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Components Manager

<Tip warning={true}>

🧪 **Experimental Feature**: This is an experimental feature we are actively developing. The API may be subject to breaking changes.

</Tip>

The Components Manager is a central model registry and management system in diffusers. It lets you add models then reuse them across multiple pipelines and workflows. It tracks all models in one place with useful metadata such as model size, device placement and loaded adapters (LoRA, IP-Adapter). It has mechanisms in place to prevent duplicate model instances, enables memory-efficient sharing. Most significantly, it offers offloading that works across pipelines — unlike regular DiffusionPipeline offloading (i.e. `enable_model_cpu_offload` and `enable_sequential_cpu_offload`) which is limited to one pipeline with predefined sequences, the Components Manager automatically manages your device memory across all your models and workflows. 


## Basic Operations

Let's start with the most basic operations. First, create a Components Manager:

```py
from diffusers import ComponentsManager
comp = ComponentsManager()
```

Use the `add(name, component)` method to register a component. It returns a unique ID that combines the component name with the object's unique identifier (using Python's `id()` function):

```py
from diffusers import AutoModel
text_encoder = AutoModel.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="text_encoder")
# Returns component_id like 'text_encoder_139917733042864'
component_id = comp.add("text_encoder", text_encoder)
```

You can view all registered components and their metadata:

```py
>>> comp
Components:
===============================================================================================================================================
Models:
-----------------------------------------------------------------------------------------------------------------------------------------------
Name_ID                      | Class                     | Device: act(exec)    | Dtype           | Size (GB)  | Load ID         | Collection
-----------------------------------------------------------------------------------------------------------------------------------------------
text_encoder_139917733042864 | CLIPTextModel             | cpu                  | torch.float32   | 0.46       | N/A             | N/A
-----------------------------------------------------------------------------------------------------------------------------------------------

Additional Component Info:
==================================================
```

And remove components using their unique ID:

```py
comp.remove("text_encoder_139917733042864")
```

## Duplicate Detection

The Components Manager automatically detects and prevents duplicate model instances to save memory and avoid confusion. Let's walk through how this works in practice.

When you try to add the same object twice, the manager will warn you and return the existing ID:

```py
>>> comp.add("text_encoder", text_encoder)
'text_encoder_139917733042864'
>>> comp.add("text_encoder", text_encoder)
ComponentsManager: component 'text_encoder' already exists as 'text_encoder_139917733042864'
'text_encoder_139917733042864'
```

Even if you add the same object under a different name, it will still be detected as a duplicate:

```py
>>> comp.add("clip", text_encoder)
ComponentsManager: adding component 'clip' as 'clip_139917733042864', but it is duplicate of 'text_encoder_139917733042864'
To remove a duplicate, call `components_manager.remove('<component_id>')`.
'clip_139917733042864'
```

However, there's a more subtle case where duplicate detection becomes tricky. When you load the same model into different objects, the manager can't detect duplicates unless you use `ComponentSpec`. For example:

```py
>>> text_encoder_2 = AutoModel.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="text_encoder")
>>> comp.add("text_encoder", text_encoder_2)
'text_encoder_139917732983664'
```

This creates a problem - you now have two copies of the same model consuming double the memory:

```py
>>> comp
Components:
===============================================================================================================================================
Models:
-----------------------------------------------------------------------------------------------------------------------------------------------
Name_ID                      | Class                     | Device: act(exec)    | Dtype           | Size (GB)  | Load ID         | Collection
-----------------------------------------------------------------------------------------------------------------------------------------------
text_encoder_139917733042864 | CLIPTextModel             | cpu                  | torch.float32   | 0.46       | N/A             | N/A
clip_139917733042864         | CLIPTextModel             | cpu                  | torch.float32   | 0.46       | N/A             | N/A
text_encoder_139917732983664 | CLIPTextModel             | cpu                  | torch.float32   | 0.46       | N/A             | N/A
-----------------------------------------------------------------------------------------------------------------------------------------------

Additional Component Info:
==================================================
```

We recommend using `ComponentSpec` to load your models. Models loaded with `ComponentSpec` get tagged with a unique ID that encodes their loading parameters, allowing the Components Manager to detect when different objects represent the same underlying checkpoint:

```py
from diffusers import ComponentSpec, ComponentsManager
from transformers import CLIPTextModel
comp = ComponentsManager()

# Create ComponentSpec for the first text encoder
spec = ComponentSpec(name="text_encoder", repo="stabilityai/stable-diffusion-xl-base-1.0", subfolder="text_encoder", type_hint=AutoModel)
# Create ComponentSpec for a duplicate text encoder (it is same checkpoint, from same repo/subfolder)
spec_duplicated = ComponentSpec(name="text_encoder_duplicated", repo="stabilityai/stable-diffusion-xl-base-1.0", subfolder="text_encoder", type_hint=CLIPTextModel)

# Load and add both components - the manager will detect they're the same model
comp.add("text_encoder", spec.load())
comp.add("text_encoder_duplicated", spec_duplicated.load())
```

Now the manager detects the duplicate and warns you:

```out
ComponentsManager: adding component 'text_encoder_duplicated_139917580682672', but it has duplicate load_id 'stabilityai/stable-diffusion-xl-base-1.0|text_encoder|null|null' with existing components: text_encoder_139918506246832. To remove a duplicate, call `components_manager.remove('<component_id>')`.
'text_encoder_duplicated_139917580682672'
```

Both models now show the same `load_id`, making it clear they're the same model:

```py
>>> comp
Components:
======================================================================================================================================================================================================
Models:
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Name_ID                                 | Class                     | Device: act(exec)    | Dtype           | Size (GB)  | Load ID                                                         | Collection
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
text_encoder_139918506246832            | CLIPTextModel             | cpu                  | torch.float32   | 0.46       | stabilityai/stable-diffusion-xl-base-1.0|text_encoder|null|null | N/A
text_encoder_duplicated_139917580682672 | CLIPTextModel             | cpu                  | torch.float32   | 0.46       | stabilityai/stable-diffusion-xl-base-1.0|text_encoder|null|null | N/A
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Additional Component Info:
==================================================
```

## Collections

Collections are labels you can assign to components for better organization and management. You add a component under a collection by passing the `collection=` parameter when you add the component to the manager, i.e. `add(name, component, collection=...)`. Within each collection, only one component per name is allowed - if you add a second component with the same name, the first one is automatically removed.

Here's how collections work in practice:

```py
comp = ComponentsManager()
# Create ComponentSpec for the first UNet (SDXL base)
spec = ComponentSpec(name="unet", repo="stabilityai/stable-diffusion-xl-base-1.0", subfolder="unet", type_hint=AutoModel)
# Create ComponentSpec for a different UNet (Juggernaut-XL)
spec2 = ComponentSpec(name="unet", repo="RunDiffusion/Juggernaut-XL-v9", subfolder="unet", type_hint=AutoModel, variant="fp16")

# Add both UNets to the same collection - the second one will replace the first
comp.add("unet", spec.load(), collection="sdxl")
comp.add("unet", spec2.load(), collection="sdxl")
```

The manager automatically removes the old UNet and adds the new one:

```out
ComponentsManager: removing existing unet from collection 'sdxl': unet_139917723891888
'unet_139917723893136'
```

Only one UNet remains in the collection:

```py
>>> comp
Components:
====================================================================================================================================================================
Models:
--------------------------------------------------------------------------------------------------------------------------------------------------------------------
Name_ID              | Class                     | Device: act(exec)    | Dtype           | Size (GB)  | Load ID                                      | Collection
--------------------------------------------------------------------------------------------------------------------------------------------------------------------
unet_139917723893136 | UNet2DConditionModel      | cpu                  | torch.float32   | 9.56       | RunDiffusion/Juggernaut-XL-v9|unet|fp16|null | sdxl
--------------------------------------------------------------------------------------------------------------------------------------------------------------------

Additional Component Info:
==================================================
```

For example, in node-based systems, you can mark all models loaded from one node with the same collection label, automatically replace models when user loads new checkpoints under same name, batch delete all models in a collection when a node is removed.

## Retrieving Components

The Components Manager provides several methods to retrieve registered components.

The `get_one()` method returns a single component and supports pattern matching for the `name` parameter. You can use:
- exact matches like `comp.get_one(name="unet")`
- wildcards like `comp.get_one(name="unet*")` for components starting with "unet"
- exclusion patterns like `comp.get_one(name="!unet")` to exclude components named "unet"
- OR patterns like `comp.get_one(name="unet|vae")` to match either "unet" OR "vae". 

Optionally, You can add collection and load_id as filters e.g. `comp.get_one(name="unet", collection="sdxl")`. If multiple components match, `get_one()` throws an error.

Another useful method is `get_components_by_names()`, which takes a list of names and returns a dictionary mapping names to components. This is particularly helpful with modular pipelines since they provide lists of required component names, and the returned dictionary can be directly passed to `pipeline.update_components()`.

```py
# Get components by name list
component_dict = comp.get_components_by_names(names=["text_encoder", "unet", "vae"])
# Returns: {"text_encoder": component1, "unet": component2, "vae": component3}
```

## Using Components Manager with Modular Pipelines

The Components Manager integrates seamlessly with Modular Pipelines. All you need to do is pass a Components Manager instance to `from_pretrained()` or `init_pipeline()` with an optional `collection` parameter:

```py
from diffusers import ModularPipeline, ComponentsManager
comp = ComponentsManager()
pipe = ModularPipeline.from_pretrained("YiYiXu/modular-demo-auto", components_manager=comp, collection="test1")
```

By default, modular pipelines don't load components immediately, so both the pipeline and Components Manager start empty:

```py
>>> comp
Components:
==================================================
No components registered.
==================================================
```

When you load components on the pipeline, they are automatically registered in the Components Manager:

```py
>>> pipe.load_components(names="unet")
>>> comp
Components:
==============================================================================================================================================================
Models:
--------------------------------------------------------------------------------------------------------------------------------------------------------------
Name_ID              | Class                     | Device: act(exec)    | Dtype           | Size (GB)  | Load ID                                | Collection
--------------------------------------------------------------------------------------------------------------------------------------------------------------
unet_139917726686304 | UNet2DConditionModel      | cpu                  | torch.float32   | 9.56       | SG161222/RealVisXL_V4.0|unet|null|null | test1
--------------------------------------------------------------------------------------------------------------------------------------------------------------

Additional Component Info:
==================================================
```

Now let's load all default components and then create a second pipeline that reuses all components from the first one. We pass the same Components Manager to the second pipeline but with a different collection:

```py
# Load all default components 
>>> pipe.load_default_components()

# Create a second pipeline using the same Components Manager but with a different collection
>>> pipe2 = ModularPipeline.from_pretrained("YiYiXu/modular-demo-auto", components_manager=comp, collection="test2")
```

As mentioned earlier, `ModularPipeline` has a property `null_component_names` that returns a list of component names it needs to load. We can conveniently use this list with the `get_components_by_names` method on the Components Manager:

```py
# Get the list of components that pipe2 needs to load
>>> pipe2.null_component_names 
['text_encoder', 'text_encoder_2', 'tokenizer', 'tokenizer_2', 'image_encoder', 'unet', 'vae', 'scheduler', 'controlnet']

# Retrieve all required components from the Components Manager
>>> comp_dict = comp.get_components_by_names(names=pipe2.null_component_names)

# Update the pipeline with the retrieved components
>>> pipe2.update_components(**comp_dict)
```

The warnings that follow are expected and indicate that the Components Manager is correctly identifying that these components already exist and will be reused rather than creating duplicates:

```out
ComponentsManager: component 'text_encoder' already exists as 'text_encoder_139917586016400'
ComponentsManager: component 'text_encoder_2' already exists as 'text_encoder_2_139917699973424'
ComponentsManager: component 'tokenizer' already exists as 'tokenizer_139917580599504'
ComponentsManager: component 'tokenizer_2' already exists as 'tokenizer_2_139915763443904'
ComponentsManager: component 'image_encoder' already exists as 'image_encoder_139917722468304'
ComponentsManager: component 'unet' already exists as 'unet_139917580609632'
ComponentsManager: component 'vae' already exists as 'vae_139917722459040'
ComponentsManager: component 'scheduler' already exists as 'scheduler_139916266559408'
ComponentsManager: component 'controlnet' already exists as 'controlnet_139917722454432'
```


The pipeline is now fully loaded:

```py
# null_component_names return empty list, meaning everything are loaded
>>> pipe2.null_component_names
[]
```

No new components were added to the Components Manager - we're reusing everything. All models are now associated with both `test1` and `test2` collections, showing that these components are shared across multiple pipelines:
```py
>>> comp
Components:
========================================================================================================================================================================================
Models:
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Name_ID                        | Class                         | Device: act(exec)    | Dtype           | Size (GB)  | Load ID                                            | Collection
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
text_encoder_139917586016400   | CLIPTextModel                 | cpu                  | torch.float32   | 0.46       | SG161222/RealVisXL_V4.0|text_encoder|null|null     | test1
                               |                               |                      |                 |            |                                                    | test2
text_encoder_2_139917699973424 | CLIPTextModelWithProjection   | cpu                  | torch.float32   | 2.59       | SG161222/RealVisXL_V4.0|text_encoder_2|null|null   | test1
                               |                               |                      |                 |            |                                                    | test2
unet_139917580609632           | UNet2DConditionModel          | cpu                  | torch.float32   | 9.56       | SG161222/RealVisXL_V4.0|unet|null|null             | test1
                               |                               |                      |                 |            |                                                    | test2
controlnet_139917722454432     | ControlNetModel               | cpu                  | torch.float32   | 4.66       | diffusers/controlnet-canny-sdxl-1.0|null|null|null | test1
                               |                               |                      |                 |            |                                                    | test2
vae_139917722459040            | AutoencoderKL                 | cpu                  | torch.float32   | 0.31       | SG161222/RealVisXL_V4.0|vae|null|null              | test1
                               |                               |                      |                 |            |                                                    | test2
image_encoder_139917722468304  | CLIPVisionModelWithProjection | cpu                  | torch.float32   | 6.87       | h94/IP-Adapter|sdxl_models/image_encoder|null|null | test1
                               |                               |                      |                 |            |                                                    | test2
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Other Components:
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
ID                             | Class                         | Collection
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
tokenizer_139917580599504      | CLIPTokenizer                 | test1
                               |                               | test2
scheduler_139916266559408      | EulerDiscreteScheduler        | test1
                               |                               | test2
tokenizer_2_139915763443904    | CLIPTokenizer                 | test1
                               |                               | test2
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Additional Component Info:
==================================================
```


## Automatic Memory Management

The Components Manager provides a global offloading strategy across all models, regardless of which pipeline is using them:

```py
comp.enable_auto_cpu_offload(device="cuda")
```

When enabled, all models start on CPU. The manager moves models to the device right before they're used and moves other models back to CPU when GPU memory runs low. You can set your own rules for which models to offload first. This works smoothly as you add or remove components. Once it's on, you don't need to worry about device placement - you can focus on your workflow.



## Practical Example: Building Modular Workflows with Component Reuse

Now that we've covered the basics of the Components Manager, let's walk through a practical example that shows how to build workflows in a modular setting and use the Components Manager to reuse components across multiple pipelines. This example demonstrates the true power of Modular Diffusers by working with multiple pipelines that can share components. 

In this example, we'll generate latents from a text-to-image pipeline, then refine them with an image-to-image pipeline.

Let's create a modular text-to-image workflow by separating it into three workflows: `text_blocks` for encoding prompts, `t2i_blocks` for generating latents, and `decoder_blocks` for creating final images.

```py
import torch
from diffusers.modular_pipelines import SequentialPipelineBlocks
from diffusers.modular_pipelines.stable_diffusion_xl import ALL_BLOCKS

# Create modular blocks and separate text encoding and decoding steps
t2i_blocks = SequentialPipelineBlocks.from_blocks_dict(ALL_BLOCKS["text2img"])
text_blocks = t2i_blocks.sub_blocks.pop("text_encoder")
decoder_blocks = t2i_blocks.sub_blocks.pop("decode")
```

Now we will convert them into runnalbe pipelines and set up the Components Manager with auto offloading and organize components under a "t2i" collection

Since we now have 3 different workflows that share components, we create a separate pipeline that serves as a dedicated loader to load all the components, register them to the component manager, and then reuse them across different workflows.

```py
from diffusers import ComponentsManager, ModularPipeline

# Set up Components Manager with auto offloading
components = ComponentsManager()
components.enable_auto_cpu_offload(device="cuda")

# Create a new pipeline to load the components
t2i_repo = "YiYiXu/modular-demo-auto"
t2i_loader_pipe = ModularPipeline.from_pretrained(t2i_repo, components_manager=components, collection="t2i")

# convert the 3 blocks into pipelines and attach the same components manager to all 3
text_node = text_blocks.init_pipeline(t2i_repo, components_manager=components)
decoder_node = decoder_blocks.init_pipeline(t2i_repo, components_manager=components)
t2i_pipe = t2i_blocks.init_pipeline(t2i_repo, components_manager=components)
```

Load all components into the loader pipeline, they should all be automatically registered to Components Manager under the "t2i" collection:

```py
# Load all components (including IP-Adapter and ControlNet for later use)
t2i_loader_pipe.load_default_components(torch_dtype=torch.float16)
```

Now distribute the loaded components to each pipeline:

```py
# Get VAE for decoder (using get_one since there's only one)
vae = components.get_one(load_id="SG161222/RealVisXL_V4.0|vae|null|null")
decoder_node.update_components(vae=vae)

# Get text components for text node (using get_components_by_names for multiple components)
text_components = components.get_components_by_names(text_node.null_component_names)
text_node.update_components(**text_components)

# Get remaining components for t2i pipeline
t2i_components = components.get_components_by_names(t2i_pipe.null_component_names)
t2i_pipe.update_components(**t2i_components)
```

Now we can generate images using our modular workflow:

```py
# Generate text embeddings
prompt = "an astronaut"
text_embeddings = text_node(prompt=prompt, output=["prompt_embeds","negative_prompt_embeds", "pooled_prompt_embeds", "negative_pooled_prompt_embeds"])

# Generate latents and decode to image
generator = torch.Generator(device="cuda").manual_seed(0)
latents_t2i = t2i_pipe(**text_embeddings, num_inference_steps=25, generator=generator, output="latents")
image = decoder_node(latents=latents_t2i, output="images")[0]
image.save("modular_part2_t2i.png")
```

Let's add a LoRA:

```py
# Load LoRA weights 
>>> t2i_loader_pipe.load_lora_weights("CiroN2022/toy-face", weight_name="toy_face_sdxl.safetensors", adapter_name="toy_face")
>>> components
Components:
============================================================================================================================================================
...
Additional Component Info:
==================================================

unet:
  Adapters: ['toy_face']
```

You can see that the Components Manager tracks adapters metadata for all models it manages, and in our case, only Unet has lora loaded. This means we can reuse existing text embeddings. 

```py
# Generate with LoRA (reusing existing text embeddings)
generator = torch.Generator(device="cuda").manual_seed(0)
latents_lora = t2i_pipe(**text_embeddings, num_inference_steps=25, generator=generator, output="latents")
image = decoder_node(latents=latents_lora, output="images")[0]
image.save("modular_part2_lora.png")
```


Now let's create a refiner pipeline that reuses components from our text-to-image workflow:

```py
# Create refiner blocks (removing image_encoder and decode since we work with latents)
refiner_blocks = SequentialPipelineBlocks.from_blocks_dict(ALL_BLOCKS["img2img"])
refiner_blocks.sub_blocks.pop("image_encoder")
refiner_blocks.sub_blocks.pop("decode")

# Create refiner pipeline with different repo and collection,
# Attach the same component manager to it
refiner_repo = "YiYiXu/modular_refiner"
refiner_pipe = refiner_blocks.init_pipeline(refiner_repo, components_manager=components, collection="refiner")
```

We pass the **same Components Manager** (`components`) to the refiner pipeline, but with a **different collection** (`"refiner"`). This allows the refiner to access and reuse components from the "t2i" collection while organizing its own components (like the refiner UNet) under the "refiner" collection. 

```py
# Load only the refiner UNet (different from t2i UNet)
refiner_pipe.load_components(names="unet", torch_dtype=torch.float16)

# Reuse components from t2i pipeline using pattern matching
reuse_components = components.search_components("text_encoder_2|scheduler|vae|tokenizer_2")
refiner_pipe.update_components(**reuse_components)
```

When we reuse components from the "t2i" collection, they automatically get added to the "refiner" collection as well. You can verify this by checking the Components Manager - you'll see components like `vae`, `scheduler`, etc. listed under both collections, indicating they're shared between workflows.

Now we can refine any of our generated latents:

```py
# Refine all our different latents
refined_latents = refiner_pipe(image_latents=latents_t2i, prompt=prompt, num_inference_steps=10, output="latents")
refined_image = decoder_node(latents=refined_latents, output="images")[0]
refined_image.save("modular_part2_t2i_refine_out.png")

refined_latents = refiner_pipe(image_latents=latents_lora, prompt=prompt, num_inference_steps=10, output="latents")
refined_image = decoder_node(latents=refined_latents, output="images")[0]
refined_image.save("modular_part2_lora_refine_out.png")
```


Here are the results from our modular pipeline examples.

#### Base Text-to-Image Generation
| Base Text-to-Image | Base Text-to-Image (Refined) |
|-------------------|------------------------------|
| ![Base T2I](https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/modular_quicktour/modular_part2_t2i.png) | ![Base T2I Refined](https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/modular_quicktour/modular_part2_t2i_refine_out.png) |

#### LoRA
| LoRA              | LoRA               (Refined) |
|-------------------|------------------------------|
| ![LoRA](https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/modular_quicktour/modular_part2_lora.png) | ![LoRA Refined](https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/modular_quicktour/modular_part2_lora_refine_out.png) |

