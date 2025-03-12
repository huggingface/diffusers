<!-- Copyright 2024 The HuggingFace Team. All rights reserved.
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

# Wan

<div class="flex flex-wrap space-x-1">
  <img alt="LoRA" src="https://img.shields.io/badge/LoRA-d8b4fe?style=flat"/>
</div>

[Wan 2.1](https://github.com/Wan-Video/Wan2.1) by the Alibaba Wan Team.

<!-- TODO(aryan): update abstract once paper is out -->

<Tip>

Make sure to check out the Schedulers [guide](../../using-diffusers/schedulers) to learn how to explore the tradeoff between scheduler speed and quality, and see the [reuse components across pipelines](../../using-diffusers/loading#reuse-a-pipeline) section to learn how to efficiently load the same components into multiple pipelines.

</Tip>

Recommendations for inference:
- VAE in `torch.float32` for better decoding quality.
- `num_frames` should be of the form `4 * k + 1`, for example `49` or `81`.
- For smaller resolution videos, try lower values of `shift` (between `2.0` to `5.0`) in the [Scheduler](https://huggingface.co/docs/diffusers/main/en/api/schedulers/flow_match_euler_discrete#diffusers.FlowMatchEulerDiscreteScheduler.shift). For larger resolution videos, try higher values (between `7.0` and `12.0`). The default value is `3.0` for Wan.

### Using a custom scheduler

Wan can be used with many different schedulers, each with their own benefits regarding speed and generation quality. By default, Wan uses the `UniPCMultistepScheduler(prediction_type="flow_prediction", use_flow_sigmas=True, flow_shift=3.0)` scheduler. You can use a different scheduler as follows:

```python
from diffusers import FlowMatchEulerDiscreteScheduler, UniPCMultistepScheduler, WanPipeline

scheduler_a = FlowMatchEulerDiscreteScheduler(shift=5.0)
scheduler_b = UniPCMultistepScheduler(prediction_type="flow_prediction", use_flow_sigmas=True, flow_shift=4.0)

pipe = WanPipeline.from_pretrained("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", scheduler=<CUSTOM_SCHEDULER_HERE>)

# or,
pipe.scheduler = <CUSTOM_SCHEDULER_HERE>
```

### Using single file loading with Wan

The `WanTransformer3DModel` and `AutoencoderKLWan` models support loading checkpoints in their original format via the `from_single_file` loading 
method. 


```python
import torch
from diffusers import WanPipeline, WanTransformer3DModel

ckpt_path = "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/blob/main/split_files/diffusion_models/wan2.1_t2v_1.3B_bf16.safetensors"
transformer = WanTransformer3DModel.from_single_file(ckpt_path, torch_dtype=torch.bfloat16)

pipe = WanPipeline.from_pretrained("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", transformer=transformer)
```

## WanPipeline

[[autodoc]] WanPipeline
  - all
  - __call__

## WanImageToVideoPipeline

[[autodoc]] WanImageToVideoPipeline
  - all
  - __call__

## WanPipelineOutput

[[autodoc]] pipelines.wan.pipeline_output.WanPipelineOutput
