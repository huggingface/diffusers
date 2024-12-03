<!--Copyright 2024 The HuggingFace Team. All rights reserved.
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

# OmniGen

[OmniGen: Unified Image Generation](https://arxiv.org/pdf/2409.11340) from BAAI, by Shitao Xiao, Yueze Wang, Junjie Zhou, Huaying Yuan, Xingrun Xing, Ruiran Yan, Chaofan Li, Shuting Wang, Tiejun Huang, Zheng Liu.

The abstract from the paper is:

*The emergence of Large Language Models (LLMs) has unified language 
generation tasks and revolutionized human-machine interaction. 
However, in the realm of image generation, a unified model capable of handling various tasks
within a single framework remains largely unexplored. In
this work, we introduce OmniGen, a new diffusion model
for unified image generation. OmniGen is characterized
by the following features: 1) Unification: OmniGen not
only demonstrates text-to-image generation capabilities but
also inherently supports various downstream tasks, such
as image editing, subject-driven generation, and visual conditional generation. 2) Simplicity: The architecture of
OmniGen is highly simplified, eliminating the need for additional plugins. Moreover, compared to existing diffusion
models, it is more user-friendly and can complete complex
tasks end-to-end through instructions without the need for
extra intermediate steps, greatly simplifying the image generation workflow. 3) Knowledge Transfer: Benefit from
learning in a unified format, OmniGen effectively transfers
knowledge across different tasks, manages unseen tasks and
domains, and exhibits novel capabilities. We also explore
the model’s reasoning capabilities and potential applications of the chain-of-thought mechanism. This work represents the first attempt at a general-purpose image generation model, and we will release our resources at https:
//github.com/VectorSpaceLab/OmniGen to foster future advancements.*

<Tip>

Make sure to check out the Schedulers [guide](../../using-diffusers/schedulers.md) to learn how to explore the tradeoff between scheduler speed and quality, and see the [reuse components across pipelines](../../using-diffusers/loading.md#reuse-a-pipeline) section to learn how to efficiently load the same components into multiple pipelines.

</Tip>

This pipeline was contributed by [staoxiao](https://github.com/staoxiao). The original codebase can be found [here](https://github.com/VectorSpaceLab/OmniGen). The original weights can be found under [hf.co/shitao](https://huggingface.co/Shitao/OmniGen-v1).


## Inference

Use [`torch.compile`](https://huggingface.co/docs/diffusers/main/en/tutorials/fast_diffusion#torchcompile) to reduce the inference latency.

First, load the pipeline:

```python
import torch
from diffusers import CogVideoXPipeline, CogVideoXImageToVideoPipeline
from diffusers.utils import export_to_video,load_image
pipe = CogVideoXPipeline.from_pretrained("THUDM/CogVideoX-5b").to("cuda") # or "THUDM/CogVideoX-2b" 
```

If you are using the image-to-video pipeline, load it as follows:

```python
pipe = CogVideoXImageToVideoPipeline.from_pretrained("THUDM/CogVideoX-5b-I2V").to("cuda")
```

Then change the memory layout of the pipelines `transformer` component to `torch.channels_last`:

```python
pipe.transformer.to(memory_format=torch.channels_last)
```

Compile the components and run inference:

```python
pipe.transformer = torch.compile(pipeline.transformer, mode="max-autotune", fullgraph=True)

# CogVideoX works well with long and well-described prompts
prompt = "A panda, dressed in a small, red jacket and a tiny hat, sits on a wooden stool in a serene bamboo forest. The panda's fluffy paws strum a miniature acoustic guitar, producing soft, melodic tunes. Nearby, a few other pandas gather, watching curiously and some clapping in rhythm. Sunlight filters through the tall bamboo, casting a gentle glow on the scene. The panda's face is expressive, showing concentration and joy as it plays. The background includes a small, flowing stream and vibrant green foliage, enhancing the peaceful and magical atmosphere of this unique musical performance."
video = pipe(prompt=prompt, guidance_scale=6, num_inference_steps=50).frames[0]
```

The [T2V benchmark](https://gist.github.com/a-r-r-o-w/5183d75e452a368fd17448fcc810bd3f) results on an 80GB A100 machine are:

```
Without torch.compile(): Average inference time: 96.89 seconds.
With torch.compile(): Average inference time: 76.27 seconds.
```


## CogVideoXPipeline

[[autodoc]] CogVideoXPipeline
  - all
  - __call__


