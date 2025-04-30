<!--Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
-->

# VisualCloze

[VisualCloze: A Universal Image Generation Framework via Visual In-Context Learning](https://arxiv.org/abs/2504.07960) is an in-context learning based universal image generation framework that can 1) support various in-domain tasks, 2) generalize to unseen tasks through in-context learning, 3) unify multiple tasks into one step and generate both target image and intermediate results, and 4) support reverse-engineering a set of conditions from a target image.

The abstract from the paper is:

*Recent progress in diffusion models significantly advances various image generation tasks. However, the current mainstream approach remains focused on building task-specific models, which have limited efficiency when supporting a wide range of different needs. While universal models attempt to address this limitation, they face critical challenges, including generalizable task instruction, appropriate task distributions, and unified architectural design. To tackle these challenges, we propose VisualCloze, a universal image generation framework, which supports a wide range of in-domain tasks, generalization to unseen ones, unseen unification of multiple tasks, and reverse generation. Unlike existing methods that rely on language-based task instruction, leading to task ambiguity and weak generalization, we integrate visual in-context learning, allowing models to identify tasks from visual demonstrations. Meanwhile, the inherent sparsity of visual task distributions hampers the learning of transferable knowledge across tasks. To this end, we introduce Graph200K, a graph-structured dataset that establishes various interrelated tasks, enhancing task density and transferable knowledge. Furthermore, we uncover that our unified image generation formulation shared a consistent objective with image infilling, enabling us to leverage the strong generative priors of pre-trained infilling models without modifying the architectures. The codes, dataset, and models are available at https://visualcloze.github.io.*

## Inference

### Model loading

VisualCloze releases two models suitable for diffusers, i.e., VisualClozePipeline-384 and VisualClozePipeline-512, which are trained with resolutions of 384 and 512, respectively. 
The resolution means that each image is resized to the area of the square of it before concatenating images into a grid layout. 
In this case, VisualCloze uses [SDEdit](https://arxiv.org/abs/2108.01073) to upsample the generated images.
```python
import torch
from diffusers import VisualClozePipeline

pipe = VisualClozePipeline.from_pretrained("VisualCloze/VisualClozePipeline-384", resolution=384, torch_dtype=torch.bfloat16)
pipe.to("cuda")
```

### Input prompts
VisualCloze supports a wide variety of tasks. You need to pass a task prompt to describe the intention of the generation task, and optionally, a content prompt to describe the caption of the image to be generated. When the content prompt is not needed, None should also be passed.

### Input images

The input image should be a List[List[Image|None]]. Excluding the last row, each row represents an in-context example. The last row represents the current query, where the image to be generated is set to None.
When using batch inference, the input images should be a List[List[List[Image|None]]], and the input prompts should be a List[str|None].

### Resolution

By default, the model first generates an image with a resolution of ${model.resolution}^2$, and then upsamples it by a factor of three. You can try setting the `upsampling_height` and `upsampling_width` parameters to generate images with different size. 


### Examples 


More examples covering a wide range of tasks can be found in the [Online Demo](https://huggingface.co/spaces/VisualCloze/VisualCloze) and [Github Repo](https://github.com/lzyhha/VisualCloze). 
Here, the document shows simple examples for mask2image, edge-detection, and subject-driven generation.

#### mask2image

```python

# Load in-context images (make sure the paths are correct and accessible)
image_paths = [
    # in-context examples
    [
        load_image('https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/visualcloze/visualcloze_mask2image_incontext-example-1_mask.jpg'),
        load_image('https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/visualcloze/visualcloze_mask2image_incontext-example-1_image.jpg'),
    ],
    # query with the target image
    [
        load_image('https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/visualcloze/visualcloze_mask2image_query_mask.jpg'),
        None, # No image needed for the target image
    ],
]

# Task and content prompt
task_prompt = "In each row, a logical task is demonstrated to achieve [IMAGE2] an aesthetically pleasing photograph based on [IMAGE1] sam 2-generated masks with rich color coding."
content_prompt = """Majestic photo of a golden eagle perched on a rocky outcrop in a mountainous landscape. 
The eagle is positioned in the right foreground, facing left, with its sharp beak and keen eyes prominently visible. 
Its plumage is a mix of dark brown and golden hues, with intricate feather details. 
The background features a soft-focus view of snow-capped mountains under a cloudy sky, creating a serene and grandiose atmosphere. 
The foreground includes rugged rocks and patches of green moss. Photorealistic, medium depth of field, 
soft natural lighting, cool color palette, high contrast, sharp focus on the eagle, blurred background, 
tranquil, majestic, wildlife photography."""

# Run the pipeline
image_result = pipe(
    task_prompt=task_prompt,
    content_prompt=content_prompt,
    image=image_paths,
    upsampling_width=1344,
    upsampling_height=768,
    upsampling_strength=0.4,
    guidance_scale=30,
    num_inference_steps=30,
    max_sequence_length=512,
    generator=torch.Generator("cpu").manual_seed(0)
).images[0][0]

# Save the resulting image
image_result.save("visualcloze.png")
```

#### Example for edge-detection

```python
# Load in-context images (make sure the paths are correct and accessible)
image_paths = [
    # in-context examples
    [
        load_image('https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/visualcloze/visualcloze_edgedetection_incontext-example-1_image.jpg'),
        load_image('https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/visualcloze/visualcloze_edgedetection_incontext-example-1_edge.jpg'),
    ],
    [
        load_image('https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/visualcloze/visualcloze_edgedetection_incontext-example-2_image.jpg'),
        load_image('https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/visualcloze/visualcloze_edgedetection_incontext-example-2_edge.jpg'),
    ],
    # query with the target image
    [
        load_image('https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/visualcloze/visualcloze_edgedetection_query_image.jpg'),
        None, # No image needed for the target image
    ],
]

# Task and content prompt
task_prompt = "Each row illustrates a pathway from [IMAGE1] a sharp and beautifully composed photograph to [IMAGE2] edge map with natural well-connected outlines using a clear logical task."
content_prompt = ""

# Run the pipeline
image_result = pipe(
    task_prompt=task_prompt,
    content_prompt=content_prompt,
    image=image_paths,
    upsampling_width=864,
    upsampling_height=1152,
    upsampling_strength=0.4,
    guidance_scale=30,
    num_inference_steps=30,
    max_sequence_length=512,
    generator=torch.Generator("cpu").manual_seed(0)
).images[0][0]

# Save the resulting image
image_result.save("visualcloze.png")
```

#### Example for subject-driven generation

```python
# Load in-context images (make sure the paths are correct and accessible)
image_paths = [
    # in-context examples
    [
        load_image('https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/visualcloze/visualcloze_subjectdriven_incontext-example-1_reference.jpg'),
        load_image('https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/visualcloze/visualcloze_subjectdriven_incontext-example-1_depth.jpg'),
        load_image('https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/visualcloze/visualcloze_subjectdriven_incontext-example-1_image.jpg'),
    ],
    [
        load_image('https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/visualcloze/visualcloze_subjectdriven_incontext-example-2_reference.jpg'),
        load_image('https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/visualcloze/visualcloze_subjectdriven_incontext-example-2_depth.jpg'),
        load_image('https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/visualcloze/visualcloze_subjectdriven_incontext-example-2_image.jpg'),
    ],
    # query with the target image
    [
        load_image('https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/visualcloze/visualcloze_subjectdriven_query_reference.jpg'),
        load_image('https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/visualcloze/visualcloze_subjectdriven_query_depth.jpg'),
        None, # No image needed for the target image
    ],
]

# Task and content prompt
task_prompt = """Each row describes a process that begins with [IMAGE1] an image containing the key object, 
[IMAGE2] depth map revealing gray-toned spatial layers and results in 
[IMAGE3] an image with artistic qualitya high-quality image with exceptional detail."""
content_prompt = """A vintage porcelain collector's item. Beneath a blossoming cherry tree in early spring, 
this treasure is photographed up close, with soft pink petals drifting through the air and vibrant blossoms framing the scene."""

# Run the pipeline
image_result = pipe(
    task_prompt=task_prompt,
    content_prompt=content_prompt,
    image=image_paths,
    upsampling_width=1024,
    upsampling_height=1024,
    upsampling_strength=0.2,
    guidance_scale=30,
    num_inference_steps=30,
    max_sequence_length=512,
    generator=torch.Generator("cpu").manual_seed(0)
).images[0][0]

# Save the resulting image
image_result.save("visualcloze.png")
```

## VisualClozePipeline

[[autodoc]] VisualClozePipeline
  - all
  - __call__

## VisualClozeGenerationPipeline

[[autodoc]] VisualClozeGenerationPipeline
  - all
  - __call__

## VisualClozeUpsamplingPipeline

[[autodoc]] VisualClozeUpsamplingPipeline
  - all
  - __call__
