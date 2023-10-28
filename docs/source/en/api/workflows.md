<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Shareable workflows

<Tip warning={true}>

🧪 Workflow is experimental and its APIs can change in the future.

</Tip>

Workflows provide a simple mechanism to share your pipeline call arguments, making it easier to reproduce results. 

## Serializing a workflow

```python
from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, safety_checker=None)
pipeline.to("cuda")

outputs = pipeline("A painting of a horse", num_inference_steps=15, return_workflow=True)
workflow = outputs.workflow
```

If you look at this specific workflow, you'll see values like the number of inference steps, guidance scale, and height and width:

```bash
{'prompt': 'A painting of a horse',
 'height': 512,
 'width': 512,
 'num_inference_steps': 15,
 'guidance_scale': 7.5,
 'negative_prompt': None,
 'num_images_per_prompt': 1,
 'eta': 0.0,
 'output_type': 'pil',
 'return_dict': True,
 'callback': None,
 'callback_steps': 1,
 'cross_attention_kwargs': None,
 'guidance_rescale': 0.0,
 'clip_skip': None,
 '_name_or_path': 'runwayml/stable-diffusion-v1-5'}
```

A [`Workflow`] object provides all the argument values in the `__call__()` of a pipeline. Add `return_workflow=True` to return a `Workflow` object:

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
<Tip>

You could also pass `prompt_embeds` instead of a `prompt`.

</Tip>
```python
image = pipeline().images[0]
```

You can also override the pipeline call arguments. For example, to add a `negative_prompt`:

```python
image = pipeline(negative_prompt="bad quality").images[0]
```

Loading from a specific workflow is possible by specifying the `filename` argument inside the [`DiffusionPipeline.load_workflow`] method.

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
 'height': 512,
 'width': 512,
 'num_inference_steps': 20,
 'guidance_scale': 7.5,
 'negative_prompt': None,
 'eta': 0.0,
 'output_type': 'pil',
 'return_dict': True,
 'callback': None,
 'callback_steps': 1,
 'cross_attention_kwargs': None,
 'controlnet_conditioning_scale': 1.0,
 'guess_mode': False,
 'control_guidance_start': [0.0],
 'control_guidance_end': [1.0],
 'clip_skip': None,
 'generator_seed': None,
 'generator_device': 'cpu',
 '_name_or_path': 'runwayml/stable-diffusion-v1-5'}
```

As you can notice, the `image` passed to the `pipeline` isn't a part of `workflow`.

Let's serialize it and reload the pipeline:

```python
workflow.save_workflow("my-simple-workflow-sd", filename="controlnet_simple.json", push_to_hub=True)
```

Steps to load the workflow into [`StableDiffusionControlNetPipeline`] should remain the same as the example above:

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

## workfllow_utils.populate_workflow_from_pipeline

[[autodoc]] workflow_utils.populate_workflow_from_pipeline