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

# Cosmos3

Cosmos3 is NVIDIA's joint text + video + sound generation pipeline built on the [Cosmos World Foundation Model Platform](https://huggingface.co/papers/2501.03575). A single Mixture-of-Transformer (`Cosmos3OmniTransformer`) runs a Qwen-style language model in parallel with a diffusion generation pathway: text tokens flow through a causal "understanding" stream while video and sound latents flow through a bi-directionally-attended "generation" stream, joined by a 3D multimodal RoPE.

Released weights live under the [`nvidia/Cosmos3-Nano`](https://huggingface.co/nvidia/Cosmos3-Nano) repo on the Hub. The same pipeline class supports text-to-image, text-to-video, image-to-video, and (with a sound-capable checkpoint) text+image-to-video-with-sound.

> [!TIP]
> Make sure to check out the Schedulers [guide](../../using-diffusers/schedulers) to learn how to explore the tradeoff between scheduler speed and quality, and see the [reuse components across pipelines](../../using-diffusers/loading#reuse-a-pipeline) section to learn how to efficiently load the same components into multiple pipelines.

## Text-to-image

Single-frame generation. The model is conditioned only on the text prompt; pass `num_frames=1`.

```python
import torch
from diffusers import Cosmos3OmniDiffusersPipeline
from diffusers.pipelines.cosmos.export_utils import save_img_or_video

pipe = Cosmos3OmniDiffusersPipeline.from_pretrained(
    "nvidia/Cosmos3-Nano", torch_dtype=torch.bfloat16, device_map="cuda"
)

prompt = (
    "A medium shot of a modern robotics research laboratory with white walls and a gray floor. "
    "A robotic arm with a metallic finish is mounted on a clean white workbench, its gripper positioned "
    "above a row of small colored objects. A laptop and neatly arranged tools sit beside the robot. "
    "A large monitor on the wall behind displays a software interface. The scene is brightly lit by "
    "overhead fluorescent lights."
)

result = pipe(prompt=prompt, num_frames=1, height=720, width=1280)
save_img_or_video(result.video[0], "cosmos3_t2i")  # writes cosmos3_t2i.jpg
```

## Text-to-video

Multi-frame generation conditioned on text alone. Pick `num_frames` based on the target duration — the default `num_frames=189` produces ≈ 7.9 s at 24 FPS.

```python
import torch
from diffusers import Cosmos3OmniDiffusersPipeline
from diffusers.pipelines.cosmos.export_utils import save_img_or_video

pipe = Cosmos3OmniDiffusersPipeline.from_pretrained(
    "nvidia/Cosmos3-Nano", torch_dtype=torch.bfloat16, device_map="cuda"
)

prompt = (
    "The video opens with a view of a well-lit indoor space featuring a wooden display case with "
    "compartments filled with various fruits, including bananas, apples, pears, oranges, and carambolas. "
    "The bananas are neatly arranged in the middle compartment, while apples are in the left and a mix "
    "of pears, oranges, and carambolas are in the right. Two robotic arms with grippers are positioned "
    "at the bottom of the frame, with the one on the left remaining stationary, partially obscuring the "
    "apples. The robotic arm on the right begins its action, extending towards the right side of the "
    "display case. It carefully picks up a pear from the fruit section, placing it into a plastic bag "
    "in the shopping cart nearby, which has red handles. After securing the pear, the arm retracts back "
    "to its original position. The process repeats as the robotic arm picks up an orange and places it "
    "in the bag, followed by a carambola. The final frame captures the robotic arm returning to its "
    "initial position, leaving the display case and surrounding area unchanged. The video showcases a "
    "seamless and efficient automated fruit-picking process, highlighting the precision and efficiency "
    "of modern robotics in a retail setting."
)

result = pipe(prompt=prompt, num_frames=189, height=720, width=1280, fps=24.0)
save_img_or_video(result.video[0], "cosmos3_t2v", fps=24)  # writes cosmos3_t2v.mp4
```

## Image-to-video

Pass a conditioning image via `image=`. The pipeline anchors frame 0 to the supplied image and denoises the rest.

```python
import torch
from diffusers import Cosmos3OmniDiffusersPipeline
from diffusers.pipelines.cosmos.export_utils import save_img_or_video
from diffusers.utils import load_image

pipe = Cosmos3OmniDiffusersPipeline.from_pretrained(
    "nvidia/Cosmos3-Nano", torch_dtype=torch.bfloat16, device_map="cuda"
)

image = load_image(
    "https://github.com/nvidia-cosmos/cosmos-dependencies/releases/download/assets/robot_153.jpg"
)
prompt = (
    "The video opens with a view of a testing environment, characterized by a large wooden table at the "
    "center. On this table, two robot arms are positioned at opposite ends, with the left arm closer to "
    "the camera and the right arm further away. Between the hands lies a dark wooden shelf with a red "
    "spherical object on its top rack, likely serving as a platform or obstacle. In the background, "
    "various pieces of equipment, including a tripod, a chair, are visible. A person wearing a blue "
    "jacket and black pants stands near the center of the room, observing the experiment, with a static "
    "hand position throughout. The floor is tiled with a patterned design, and additional items like a "
    "small robot figure and some cables can be seen scattered around the space. As the video progresses, "
    "the right robotic hand extends outward, moving from its initial position towards the red spherical "
    "object on the shelf. The hand then picks up the object and places it on the lowest rack of the "
    "shelf, completing a smooth, deliberate manipulation. The left robotic hand remains stationary "
    "throughout the sequence. No new objects appear in the video; all existing elements maintain their "
    "positions except for the movement of the right robotic hand. The scene concludes with the right "
    "robotic hand returning to its initial position, while the left hand continues to rest on the table. "
    "The overall environment remains unchanged, with the focus remaining on the interaction between the "
    "robotic hands and the wooden block, highlighting precise control during the demonstration."
)

result = pipe(prompt=prompt, image=image, num_frames=189, height=720, width=1280, fps=24.0)
save_img_or_video(result.video[0], "cosmos3_i2v", fps=24)  # writes cosmos3_i2v.mp4
```

## Text-to-video with sound

When the checkpoint carries a `sound_tokenizer`, pass `enable_sound=True` to jointly generate a synchronized audio track. The waveform is returned alongside the video and decoded by the audio tokenizer's vocoder.

This is the same call as the text-to-video example above with `enable_sound=True` added:

```python
import torch
from diffusers import Cosmos3OmniDiffusersPipeline
from diffusers.pipelines.cosmos.export_utils import save_img_or_video, save_wav

pipe = Cosmos3OmniDiffusersPipeline.from_pretrained(
    "nvidia/Cosmos3-Nano", torch_dtype=torch.bfloat16, device_map="cuda"
)

prompt = (
    "The video opens with a view of a well-lit indoor space featuring a wooden display case with "
    "compartments filled with various fruits, including bananas, apples, pears, oranges, and carambolas. "
    "The bananas are neatly arranged in the middle compartment, while apples are in the left and a mix "
    "of pears, oranges, and carambolas are in the right. Two robotic arms with grippers are positioned "
    "at the bottom of the frame, with the one on the left remaining stationary, partially obscuring the "
    "apples. The robotic arm on the right begins its action, extending towards the right side of the "
    "display case. It carefully picks up a pear from the fruit section, placing it into a plastic bag "
    "in the shopping cart nearby, which has red handles. After securing the pear, the arm retracts back "
    "to its original position. The process repeats as the robotic arm picks up an orange and places it "
    "in the bag, followed by a carambola. The final frame captures the robotic arm returning to its "
    "initial position, leaving the display case and surrounding area unchanged. The video showcases a "
    "seamless and efficient automated fruit-picking process, highlighting the precision and efficiency "
    "of modern robotics in a retail setting."
)

result = pipe(
    prompt=prompt,
    num_frames=189,
    height=720,
    width=1280,
    fps=24.0,
    enable_sound=True,
)

save_img_or_video(result.video[0], "cosmos3_with_sound", fps=24)
save_wav(result.sound[0], "cosmos3_with_sound.wav", pipe.sound_tokenizer.sample_rate)
```

## Cosmos3OmniDiffusersPipeline

[[autodoc]] Cosmos3OmniDiffusersPipeline
  - all
  - __call__

## Cosmos3OmniPipelineOutput

[[autodoc]] pipelines.cosmos.pipeline_cosmos3_omni.Cosmos3OmniPipelineOutput
