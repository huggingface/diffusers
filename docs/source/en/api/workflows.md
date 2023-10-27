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

Workflows are experimental at the moment 🧪 Its APIs can change in future.

</Tip>

Workflows provide a simple mechanism to share your pipeline call arguments with others. It makes reproducibility of results easier. This doc shows you how you can leverage workflows for different use cases. 

## Serializing a workflow

```python
from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, safety_checker=None)
pipeline.to("cuda")

outputs = pipeline("A painting of a horse", num_inference_steps=15, return_workflow=True)
workflow = outputs.workflow
```

As you can notice, by specifying `return_workflow=True` in the pipeline call, you can obtain a workflow that looks like so:

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

`workflow` is a [`Workflow`] object. It provides the values of all the arguments present in the `__call__()` of a pipeline.

Once you have generated a workflow object, you can serialize it like so:

```python
outputs.workflow.save_workflow("my-simple-workflow-sd")
```

By default, your workflows will be saved with the following name: `diffusion_workflow.json`. But you can customize it by specifying the `filename` argument:

```python
outputs.workflow.save_workflow("my-simple-workflow-sd", filename="my_workflow.json")
```

By specifying `push_to_hub=True` in [`Workflow.save_workflow`], you can directly push the workflow object to the Hub. 

## Loading a workflow

You can load a workflow in a pipeline like so:

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

You can also override the pipeline call arguments. In the above example, you didn't specify any `negative_prompt`. You can easily it like so:

```python
image = pipeline(negative_prompt="bad quality").images[0]
```

Loading from a specific workflow is possible by specifying the `filename` argument inside the [`DiffusionPipeline.load_workflow`] method.

## Some common gotchas

Image-to-image pipelines like [`StableDiffusionControlNetPipeline`] accept one or more images in their calls. Currently, workflows don't support the serialization of call arguments that are of type `PIL.Image.Image` or `List[PIL.Image.Image]`. To make those pipelines work with workflows, you will have to manually pass the images. 

Let's say you have generated a workflow with the following:

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

`workflow` here looks like so:

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

Let's serialize it:

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

If you try to do `image = pipe().images[0]`, it will lead the following error:

```bash
TypeError: image must be passed and be one of PIL image, numpy array, torch tensor, list of PIL images, list of numpy arrays or list of torch tensors, but is <class 'NoneType'>
```

To resolve the error, you can just pass the conditioning image `canny_image`:

```python
image = pipe(image=canny_image).images[0]
```

## Known limitations

* We don't serialize any information on the LoRA checkpoints that might be loaded into a pipeline. So, workflows generated from pipelines loaded with a LoRA checkpoint should be handled with caution. As such, users should ensure that the respective LoRA checkpoint is first loaded into the pipeline before the corresponding workflow is loaded into the pipeline.
* Instead of passing a `prompt`, users can provide `prompt_embeds` while calling a pipeline. Currently, workflows don't serialize any call arguments that are of the following types: `torch.Tensor`, `np.ndarray`, `Callable`, `PIL.Image.Image`, and `List[PIL.Image.Image]`. 

## Workflow

[[autodoc]] workflow_utils.Workflow

## workfllow_utils.populate_workflow_from_pipeline

[[autodoc]] workflow_utils.populate_workflow_from_pipeline