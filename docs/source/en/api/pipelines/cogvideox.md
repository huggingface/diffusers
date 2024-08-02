<!-- # Copyright 2024 The HuggingFace Team. All rights reserved.
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

# CogVideoX


[The paper is still being written]() from Tsinghua University & ZhipuAI.

The abstract from the paper is:

The paper is still being written.

<Tip>

Make sure to check out the Schedulers [guide](../../using-diffusers/schedulers.md) to learn how to explore the tradeoff between scheduler speed and quality, and see the [reuse components across pipelines](../../using-diffusers/loading.md#reuse-a-pipeline) section to learn how to efficiently load the same components into multiple pipelines.

</Tip>

### Inference

Use [`torch.compile`](https://huggingface.co/docs/diffusers/main/en/tutorials/fast_diffusion#torchcompile) to reduce the inference latency.

First, load the pipeline:

```python
import torch
from diffusers import LattePipeline

pipeline = LattePipeline.from_pretrained(
	"THUDM/CogVideoX-2b", torch_dtype=torch.float16
).to("cuda")
```

Then change the memory layout of the pipelines `transformer` and `vae` components to `torch.channels-last`:

```python
pipeline.transformer.to(memory_format=torch.channels_last)
pipeline.vae.to(memory_format=torch.channels_last)
```

Finally, compile the components and run inference:

```python
pipeline.transformer = torch.compile(pipeline.transformer)
pipeline.vae.decode = torch.compile(pipeline.vae.decode)

video = pipeline(prompt="A dog wearing sunglasses floating in space, surreal, nebulae in background").frames[0]
```

The [benchmark]() results on an 80GB A100 machine are:

```
Without torch.compile(): Average inference time: 16.246 seconds.
With torch.compile(): Average inference time: 14.573 seconds.
```

## CogVideoXPipeline

[[autodoc]] CogVideoXPipeline
  - all
  - __call__

## CogVideoXPipelineOutput
[[autodoc]] pipelines.pipline_cogvideo.pipeline_output.CogVideoXPipelineOutput

