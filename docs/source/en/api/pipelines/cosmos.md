<!-- Copyright 2025 The HuggingFace Team. All rights reserved.
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
# limitations under the License. -->

# Cosmos

[Cosmos World Foundation Model Platform for Physical AI](https://huggingface.co/papers/2501.03575) by NVIDIA.

*Physical AI needs to be trained digitally first. It needs a digital twin of itself, the policy model, and a digital twin of the world, the world model. In this paper, we present the Cosmos World Foundation Model Platform to help developers build customized world models for their Physical AI setups. We position a world foundation model as a general-purpose world model that can be fine-tuned into customized world models for downstream applications. Our platform covers a video curation pipeline, pre-trained world foundation models, examples of post-training of pre-trained world foundation models, and video tokenizers. To help Physical AI builders solve the most critical problems of our society, we make our platform open-source and our models open-weight with permissive licenses available via https://github.com/NVIDIA/Cosmos.*

> [!TIP]
> Make sure to check out the Schedulers [guide](../../using-diffusers/schedulers) to learn how to explore the tradeoff between scheduler speed and quality, and see the [reuse components across pipelines](../../using-diffusers/loading#reuse-a-pipeline) section to learn how to efficiently load the same components into multiple pipelines.

## Basic usage

```python
import torch
from diffusers import Cosmos2_5_PredictBasePipeline
from diffusers.utils import export_to_video

model_id = "nvidia/Cosmos-Predict2.5-2B"
pipe = Cosmos2_5_PredictBasePipeline.from_pretrained(
    model_id, revision="diffusers/base/post-trained", torch_dtype=torch.bfloat16
)
pipe.to("cuda")

prompt = "As the red light shifts to green, the red bus at the intersection begins to move forward, its headlights cutting through the falling snow. The snowy tire tracks deepen as the vehicle inches ahead, casting fresh lines onto the slushy road. Around it, streetlights glow warmer, illuminating the drifting flakes and wet reflections on the asphalt. Other cars behind start to edge forward, their beams joining the scene. The stillness of the urban street transitions into motion as the quiet snowfall is punctuated by the slow advance of traffic through the frosty city corridor."
negative_prompt = "The video captures a series of frames showing ugly scenes, static with no motion, motion blur, over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, jerky movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special effects, fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and flickering. Overall, the video is of poor quality."

output = pipe(
    image=None,
    video=None,
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_frames=93,
    generator=torch.Generator().manual_seed(1),
).frames[0]
export_to_video(output, "text2world.mp4", fps=16)
```

## Cosmos2_5_TransferPipeline

[[autodoc]] Cosmos2_5_TransferPipeline
  - all
  - __call__


## Cosmos2_5_PredictBasePipeline

[[autodoc]] Cosmos2_5_PredictBasePipeline
  - all
  - __call__


## CosmosTextToWorldPipeline

[[autodoc]] CosmosTextToWorldPipeline
  - all
  - __call__

## CosmosVideoToWorldPipeline

[[autodoc]] CosmosVideoToWorldPipeline
  - all
  - __call__

## Cosmos2TextToImagePipeline

[[autodoc]] Cosmos2TextToImagePipeline
  - all
  - __call__

## Cosmos2VideoToWorldPipeline

[[autodoc]] Cosmos2VideoToWorldPipeline
  - all
  - __call__

## CosmosPipelineOutput

[[autodoc]] pipelines.cosmos.pipeline_output.CosmosPipelineOutput

## CosmosImagePipelineOutput

[[autodoc]] pipelines.cosmos.pipeline_output.CosmosImagePipelineOutput
