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
# limitations under the License.
-->

# Mochi 1 Preview

[Mochi 1 Preview](https://huggingface.co/genmo/mochi-1-preview) from Genmo.

*Mochi 1 preview is an open state-of-the-art video generation model with high-fidelity motion and strong prompt adherence in preliminary evaluation. This model dramatically closes the gap between closed and open video generation systems. The model is released under a permissive Apache 2.0 license.*

<Tip>

Make sure to check out the Schedulers [guide](../../using-diffusers/schedulers.md) to learn how to explore the tradeoff between scheduler speed and quality, and see the [reuse components across pipelines](../../using-diffusers/loading.md#reuse-a-pipeline) section to learn how to efficiently load the same components into multiple pipelines.

</Tip>

## Generating videos with Mochi-1 Preview

The following example will download the full precision `mochi-1-preview` weights and produce the highest quality results but will require at least 42GB VRAM to run.   

```python
import torch
from diffusers import MochiPipeline
from diffusers.utils import export_to_video

pipe = MochiPipeline.from_pretrained("genmo/mochi-1-preview")

# Enable memory savings
pipe.enable_model_cpu_offload()
pipe.enable_vae_tiling()

prompt = "Close-up of a chameleon's eye, with its scaly skin changing color. Ultra high resolution 4k."

with torch.autocast("cuda", torch.bfloat16, cache_enabled=False):
      frames = pipe(prompt, num_frames=84).frames[0]

export_to_video(frames, "mochi.mp4", fps=30)
```

## Using a lower precision variant to save memory

The following example will use the `bfloat16` variant of the model and requires 22GB VRAM to run. There is a slight drop in the quality of the generated video as a result.

```python
import torch
from diffusers import MochiPipeline
from diffusers.utils import export_to_video

pipe = MochiPipeline.from_pretrained("genmo/mochi-1-preview", variant="bf16", torch_dtype=torch.bfloat16)

# Enable memory savings
pipe.enable_model_cpu_offload()
pipe.enable_vae_tiling()

prompt = "Close-up of a chameleon's eye, with its scaly skin changing color. Ultra high resolution 4k."
frames = pipe(prompt, num_frames=84).frames[0]

export_to_video(frames, "mochi.mp4", fps=30)
```

## MochiPipeline

[[autodoc]] MochiPipeline
  - all
  - __call__

## MochiPipelineOutput

[[autodoc]] pipelines.mochi.pipeline_output.MochiPipelineOutput
