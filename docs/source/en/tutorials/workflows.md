<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Working with workflows

<Tip warning={true}>

🧪 Workflow is experimental and its APIs can change in the future.

</Tip>

Workflows provide a simple mechanism to share your pipeline call arguments and scheduler configuration, making it easier to reproduce results. 

## Serializing a workflow

A [`Workflow`] object provides all the argument values in the `__call__()` of a pipeline. Add `return_workflow=True` to return a `Workflow` object. 

```python
from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, safety_checker=None
).to("cuda")

outputs = pipeline("A painting of a horse", num_inference_steps=15, return_workflow=True)
workflow = outputs.workflow
```

If you look at this specific workflow, you'll see values like the number of inference steps, guidance scale, and height and width as well as the scheduler details:

```bash
{'prompt': 'A painting of a horse',
 'height': None,
 'width': None,
 'num_inference_steps': 15,
 'guidance_scale': 7.5,
 'negative_prompt': None,
 'eta': 0.0,
 'latents': None,
 'prompt_embeds': None,
 'negative_prompt_embeds': None,
 'output_type': 'pil',
 'return_dict': True,
 'callback': None,
 'callback_steps': 1,
 'cross_attention_kwargs': None,
 'guidance_rescale': 0.0,
 'clip_skip': None,
 'generator_seed': 331018828,
 'generator_device': device(type='cpu'),
 '_name_or_path': 'runwayml/stable-diffusion-v1-5',
 'scheduler_config': FrozenDict([('num_train_timesteps', 1000),
             ('beta_start', 0.00085),
             ('beta_end', 0.012),
             ('beta_schedule', 'scaled_linear'),
             ('trained_betas', None),
             ('skip_prk_steps', True),
             ('set_alpha_to_one', False),
             ('prediction_type', 'epsilon'),
             ('timestep_spacing', 'leading'),
             ('steps_offset', 1),
             ('_use_default_values', ['prediction_type', 'timestep_spacing']),
             ('_class_name', 'PNDMScheduler'),
             ('_diffusers_version', '0.6.0'),
             ('clip_sample', False)])}
```

Once you have generated a workflow object, you can serialize it with [`~Workflow.save_workflow`]:

```python
outputs.workflow.save_workflow("my-simple-workflow-sd")
```

By default, your workflows are saved as `diffusion_workflow.json`, but you can give them a specific name with the `filename` argument:

```python
outputs.workflow.save_workflow("my-simple-workflow-sd", filename="my_workflow.json")
```

You can also set `push_to_hub=True` in [`~Workflow.save_workflow`] to directly push the workflow object to the Hub. 

## Loading a workflow

You can load a workflow in a pipeline with [`~DiffusionPipeline.load_workflow`]:

```python
from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
).to("cuda")

pipeline.load_workflow("sayakpaul/my-simple-workflow-sd")
```

Once the pipeline is loaded with the desired workflow, it's ready to be called:

```python
image = pipeline().images[0]
```

By default, while loading a workflow, the scheduler of the underlying pipeline from the workflow isn't modified but you can change it by adding `load_scheduler=True`:

```
pipeline.load_workflow("sayakpaul/my-simple-workflow-sd", load_scheduler=True)
```

This is particularly useful if you have changed the scheduler after loading a pipeline.

You can also override the pipeline call arguments. For example, to add a `negative_prompt`:

```python
image = pipeline(negative_prompt="bad quality").images[0]
```

Loading from a workflow is possible by specifying the `filename` argument inside the [`DiffusionPipeline.load_workflow`] method.

A workflow doesn't necessarily have to be used with the same pipeline that generated it. You can use it with a different pipeline too:

```python
from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
).to("cuda")

pipeline.load_workflow("sayakpaul/my-simple-workflow-sd")
image = pipeline().images[0]
```

However, make sure to thoroughly inspect the values you are calling the pipeline with, in this case.

Loading from a local workflow is also possible:

```python
from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
).to("cuda")

pipeline.load_workflow("path_to_local_dir")
image = pipeline().images[0]
```

Alternatively, if you want to load a workflow file and populate the pipeline arguments manually:

```python
from diffusers import DiffusionPipeline
import json
import torch

pipeline = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
).to("cuda")

with open("path_to_workflow_file.json") as f:
    workflow = json.load(f)

pipeline.load_workflow(workflow)
images = pipeline().images[0]
```

## Unsupported serialization types

Image-to-image pipelines like [`StableDiffusionControlNetPipeline`] accept one or more images in their `call` method. Currently, workflows don't support serializing `call` arguments that are of type `PIL.Image.Image` or `List[PIL.Image.Image]`. To make those pipelines work with workflows, you need to pass the images manually. 

Let's say you generated the workflow below:

```python
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
import numpy as np
import torch

import cv2
from PIL import Image

# download an image
image = load_image(
    "https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png"
)
image = np.array(image)

# get canny image
image = cv2.Canny(image, 100, 200)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
canny_image = Image.fromarray(image)

# load control net and stable diffusion v1-5
controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
)
pipe.enable_model_cpu_offload()

# generate image
generator = torch.manual_seed(0)
outputs = pipe(
    prompt="futuristic-looking office", 
    image=canny_image,
    num_inference_steps=20, 
    generator=generator, 
    return_workflow=True
)
workflow = outputs.workflow
```

If you look at the workflow, you'll see the image that was passed to the pipeline isn't included:

```bash
{'prompt': 'futuristic-looking office',
 'height': None,
 'width': None,
 'num_inference_steps': 20,
 'guidance_scale': 7.5,
 'negative_prompt': None,
 'eta': 0.0,
 'latents': None,
 'prompt_embeds': None,
 'negative_prompt_embeds': None,
 'output_type': 'pil',
 'return_dict': True,
 'callback': None,
 'callback_steps': 1,
 'cross_attention_kwargs': None,
 'controlnet_conditioning_scale': 1.0,
 'guess_mode': False,
 'control_guidance_start': 0.0,
 'control_guidance_end': 1.0,
 'clip_skip': None,
 'generator_seed': 0,
 'generator_device': 'cpu',
 '_name_or_path': 'runwayml/stable-diffusion-v1-5',
 'scheduler_config': FrozenDict([('num_train_timesteps', 1000),
             ('beta_start', 0.00085),
             ('beta_end', 0.012),
             ('beta_schedule', 'scaled_linear'),
             ('trained_betas', None),
             ('solver_order', 2),
             ('prediction_type', 'epsilon'),
             ('thresholding', False),
             ('dynamic_thresholding_ratio', 0.995),
             ('sample_max_value', 1.0),
             ('predict_x0', True),
             ('solver_type', 'bh2'),
             ('lower_order_final', True),
             ('disable_corrector', []),
             ('solver_p', None),
             ('use_karras_sigmas', False),
             ('timestep_spacing', 'linspace'),
             ('steps_offset', 1),
             ('_use_default_values',
              ['lower_order_final',
               'sample_max_value',
               'solver_p',
               'dynamic_thresholding_ratio',
               'thresholding',
               'solver_type',
               'prediction_type',
               'predict_x0',
               'use_karras_sigmas',
               'disable_corrector',
               'timestep_spacing',
               'solver_order']),
             ('skip_prk_steps', True),
             ('set_alpha_to_one', False),
             ('_class_name', 'PNDMScheduler'),
             ('_diffusers_version', '0.6.0'),
             ('clip_sample', False)])}
```


Let's serialize the workflow and reload the pipeline to see what happens when you try to use it.

```python
workflow.save_workflow("my-simple-workflow-sd", filename="controlnet_simple.json", push_to_hub=True)
```

Then load the workflow into [`StableDiffusionControlNetPipeline`]:

```python
# load control net and stable diffusion v1-5
controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
)
pipe.enable_model_cpu_offload()

pipe.load_workflow("sayakpaul/my-simple-workflow-sd", filename="controlnet_simple.json")
```

If you try to generate an image now, it'll return the following error:

```bash
TypeError: image must be passed and be one of PIL image, numpy array, torch tensor, list of PIL images, list of numpy arrays or list of torch tensors, but is <class 'NoneType'>
```

To resolve the error, manually pass the conditioning image `canny_image`:

```python
image = pipe(image=canny_image).images[0]
```

Other unsupported serialization types include:

* LoRA checkpoints: any information from LoRA checkpoints that might be loaded into a pipeline isn't serialized. Workflows generated from pipelines loaded with a LoRA checkpoint should be handled cautiously! You should ensure the LoRA checkpoint is loaded into the pipeline first before loading the corresponding workflow.
* Call arguments including the following types: `torch.Tensor`, `np.ndarray`, `Callable`, `PIL.Image.Image`, and `List[PIL.Image.Image]`. 

## Workflow

[[autodoc]] workflow_utils.Workflow

## workflow_utils.populate_workflow_from_pipeline

[[autodoc]] workflow_utils.populate_workflow_from_pipeline