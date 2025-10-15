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

## Loading original format checkpoints

Original format checkpoints that have not been converted to diffusers-expected format can be loaded using the `from_single_file` method.

```python
import torch
from diffusers import Cosmos2TextToImagePipeline, CosmosTransformer3DModel

model_id = "nvidia/Cosmos-Predict2-2B-Text2Image"
transformer = CosmosTransformer3DModel.from_single_file(
    "https://huggingface.co/nvidia/Cosmos-Predict2-2B-Text2Image/blob/main/model.pt",
    torch_dtype=torch.bfloat16,
).to("cuda")
pipe = Cosmos2TextToImagePipeline.from_pretrained(model_id, transformer=transformer, torch_dtype=torch.bfloat16)
pipe.to("cuda")

prompt = "A close-up shot captures a vibrant yellow scrubber vigorously working on a grimy plate, its bristles moving in circular motions to lift stubborn grease and food residue. The dish, once covered in remnants of a hearty meal, gradually reveals its original glossy surface. Suds form and bubble around the scrubber, creating a satisfying visual of cleanliness in progress. The sound of scrubbing fills the air, accompanied by the gentle clinking of the dish against the sink. As the scrubber continues its task, the dish transforms, gleaming under the bright kitchen lights, symbolizing the triumph of cleanliness over mess."
negative_prompt = "The video captures a series of frames showing ugly scenes, static with no motion, motion blur, over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, jerky movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special effects, fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and flickering. Overall, the video is of poor quality."

output = pipe(
    prompt=prompt, negative_prompt=negative_prompt, generator=torch.Generator().manual_seed(1)
).images[0]
output.save("output.png")
```

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
