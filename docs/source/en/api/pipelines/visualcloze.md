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

Note: More examples can be found in the [Online Demo](https://huggingface.co/spaces/VisualCloze/VisualCloze) and [Github Repo](https://github.com/lzyhha/VisualCloze).

First, load the pipeline. 

Note that VisualCloze releases two models suitable for diffusers, i.e., VisualClozePipeline-384 and VisualClozePipeline-512, which are trained with resolutions of 384 and 512, respectively. 
The resolution means that each image is resized to the area of the square of it before concatenating images into a grid layout. 
In this case, VisualCloze uses [SDEdit](https://arxiv.org/abs/2108.01073) to upsample the generated images.
```python
import torch
from diffusers import VisualClozePipeline

pipe = VisualClozePipeline.from_pretrained("VisualCloze/VisualClozePipeline-384", resolution=384, torch_dtype=torch.bfloat16)
# pipe = VisualClozePipeline.from_pretrained("VisualCloze/VisualClozePipeline-512", resolution=512, torch_dtype=torch.bfloat16)
pipe.to("cuda")
```

### Input prompts
VisualCloze supports a wide variety of tasks. You need to pass a task prompt to describe the intention of the generation task, and optionally, a content prompt to describe the caption of the image to be generated. When the content prompt is not needed, None should also be passed.

### Input images

The input image should be a List[List[Image|None]]. Excluding the last row, each row represents an in-context example. The last row represents the current query, where the image to be generated is set to None.
When using batch inference, the input images should be a List[List[List[Image|None]]], and the input prompts should be a List[str|None].

### Resolution

By default, the model first generates an image with a resolution of ${model.resolution}^2$, and then upsamples it by a factor of three. You can try setting the `upsampling_height` and `upsampling_width` parameters to generate images with different size. 


```python
# Load in-context images (make sure the paths are correct and accessible)
image_paths = [
    # in-context examples
    [
        load_image('https://github.com/lzyhha/VisualCloze/raw/main/examples/examples/93bc1c43af2d6c91ac2fc966bf7725a2/93bc1c43af2d6c91ac2fc966bf7725a2_depth-anything-v2_Large.jpg'), 
        load_image('https://github.com/lzyhha/VisualCloze/raw/main/examples/examples/93bc1c43af2d6c91ac2fc966bf7725a2/93bc1c43af2d6c91ac2fc966bf7725a2.jpg'),
    ],
    # query with the target image
    [
        load_image('https://github.com/lzyhha/VisualCloze/raw/main/examples/examples/79f2ee632f1be3ad64210a641c4e201b/79f2ee632f1be3ad64210a641c4e201b_depth-anything-v2_Large.jpg'),
        None,  # No image needed for the query in this case
    ],
]

# Task and content prompt
task_prompt = "Each row outlines a logical process, starting from [IMAGE1] gray-based depth map with detailed object contours, to achieve [IMAGE2] an image with flawless clarity."
content_prompt = """A serene portrait of a young woman with long dark hair, wearing a beige dress with intricate 
gold embroidery, standing in a softly lit room. She holds a large bouquet of pale pink roses in a black box, 
positioned in the center of the frame. The background features a tall green plant to the left and a framed artwork 
on the wall to the right. A window on the left allows natural light to gently illuminate the scene. 
The woman gazes down at the bouquet with a calm expression. Soft natural lighting, warm color palette, 
high contrast, photorealistic, intimate, elegant, visually balanced, serene atmosphere."""

image_result = pipe(
    task_prompt=task_prompt,
    content_prompt=content_prompt,
    image=image_paths,
    upsampling_width=1024,
    upsampling_height=1024,
    upsampling_strength=0.4,
    guidance_scale=30,
    num_inference_steps=30,
    max_sequence_length=512,
    generator=torch.Generator("cpu").manual_seed(0)
).images[0][0]
image.save("output.png")
```

## VisualClozePipeline

[[autodoc]] VisualClozePipeline
    - all
	- __call__
