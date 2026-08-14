<!-- Copyright 2026 The HuggingFace Team. All rights reserved.
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

# Wan-Animate-2

[Wan-Animate-2](https://github.com/Wan-Video/Wan2.2) by the Alibaba Wan Team animates a reference character image with the motion of a driving video. The driving video is processed in fixed-length segments: each segment runs a reference-extraction pass that caches the driving segment's K/V in every transformer layer, denoises against that cache, and is decoded inside the loop because the next segment conditions on the previous segment's decoded tail frames.

Two presets are available: the base checkpoint samples with classifier-free guidance, and the distilled checkpoint samples in few steps without it (its guider is pinned to `guidance_scale=1.0`).

```python
import torch
from diffusers import ModularPipeline
from diffusers.utils import export_to_video, load_image, load_video

pipe = ModularPipeline.from_pretrained("Wan-AI/Wan2.2-Animate-2-14B-Diffusers")
pipe.load_components(dtype=torch.bfloat16)

# The transformer weights and the per-segment reference KV cache do not co-reside on one 80 GB
# card at the default resolution, so stream the transformer's blocks. Compiling the blocks is
# required as the in-context attention runs on the flex backend
pipe.transformer.enable_group_offload(
    onload_device=torch.device("cuda"),
    offload_device=torch.device("cpu"),
    offload_type="block_level",
    use_stream=True,
)
pipe.text_encoder.to("cuda")
pipe.image_encoder.to("cuda")
pipe.vae.to("cuda")
pipe.transformer.compile_repeated_blocks(fullgraph=False)

# The first demo from the official repository: https://github.com/Wan-Video/Wan-Animate-2
demo = "https://raw.githubusercontent.com/Wan-Video/Wan-Animate-2/main/examples/demo1"
image = load_image(f"{demo}/reference.png")
driving_video, driving_video_fps = load_video(f"{demo}/template.mp4", return_fps=True)
prompt = "人物外观描述：一只银灰色虎斑纹的小猫，拥有圆润的脸庞、竖立的耳朵和巨大的圆形眼睛。它身穿一套深蓝色的制服套装，包括一件带有金色纽扣的西装外套和一条百褶裙。外套里面搭配着白色衬衫，领口处系着一个红色的蝴蝶结，袖口露出白色的衬衫边缘。背景描述：背景为纯白色，光线均匀明亮，无其他杂物或装饰。"

videos = pipe(
    image=image,
    driving_video=driving_video,
    driving_video_fps=driving_video_fps,
    prompt=prompt,
    output="videos",
)
export_to_video(videos[0], "output.mp4", fps=24)
```

For the distilled checkpoint, load `Wan-AI/Wan2.2-Animate-2-14B-Distilled-Diffusers` the same way — nothing else changes. Each preset carries its own sampling defaults (40 steps for the base checkpoint, 10 for the distilled one), and no `guidance_scale` argument exists anywhere: guidance is owned by the pipeline's guider component (classifier-free guidance at 3.0 for the base preset, disabled for the distilled one).

`height` and `width` (defaults 800 and 640) set the target *area* of the generated video; the actual frame size keeps the reference image's aspect ratio, and the driving frames are letterboxed to it. Inputs that already sit at the target letterbox size pass through the preprocessing untouched, so preprocessing can also be done entirely outside the pipeline.

## WanAnimate2ModularPipeline

[[autodoc]] WanAnimate2ModularPipeline

## WanAnimate2DistilledModularPipeline

[[autodoc]] WanAnimate2DistilledModularPipeline

## WanAnimate2Blocks

[[autodoc]] WanAnimate2Blocks

## WanAnimate2DistilledBlocks

[[autodoc]] WanAnimate2DistilledBlocks
