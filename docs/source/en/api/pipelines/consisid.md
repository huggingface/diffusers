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

# ConsisID

<div class="flex flex-wrap space-x-1">
  <img alt="LoRA" src="https://img.shields.io/badge/LoRA-d8b4fe?style=flat"/>
</div>

[Identity-Preserving Text-to-Video Generation by Frequency Decomposition](https://huggingface.co/papers/2411.17440) from Peking University & University of Rochester & etc, by Shenghai Yuan, Jinfa Huang, Xianyi He, Yunyang Ge, Yujun Shi, Liuhan Chen, Jiebo Luo, Li Yuan.

The abstract from the paper is:

*Identity-preserving text-to-video (IPT2V) generation aims to create high-fidelity videos with consistent human identity. It is an important task in video generation but remains an open problem for generative models. This paper pushes the technical frontier of IPT2V in two directions that have not been resolved in the literature: (1) A tuning-free pipeline without tedious case-by-case finetuning, and (2) A frequency-aware heuristic identity-preserving Diffusion Transformer (DiT)-based control scheme. To achieve these goals, we propose **ConsisID**, a tuning-free DiT-based controllable IPT2V model to keep human-**id**entity **consis**tent in the generated video. Inspired by prior findings in frequency analysis of vision/diffusion transformers, it employs identity-control signals in the frequency domain, where facial features can be decomposed into low-frequency global features (e.g., profile, proportions) and high-frequency intrinsic features (e.g., identity markers that remain unaffected by pose changes). First, from a low-frequency perspective, we introduce a global facial extractor, which encodes the reference image and facial key points into a latent space, generating features enriched with low-frequency information. These features are then integrated into the shallow layers of the network to alleviate training challenges associated with DiT. Second, from a high-frequency perspective, we design a local facial extractor to capture high-frequency details and inject them into the transformer blocks, enhancing the model's ability to preserve fine-grained features. To leverage the frequency information for identity preservation, we propose a hierarchical training strategy, transforming a vanilla pre-trained video generation model into an IPT2V model. Extensive experiments demonstrate that our frequency-aware heuristic scheme provides an optimal control solution for DiT-based models. Thanks to this scheme, our **ConsisID** achieves excellent results in generating high-quality, identity-preserving videos, making strides towards more effective IPT2V. The model weight of ConsID is publicly available at https://github.com/PKU-YuanGroup/ConsisID.*

> [!TIP]
> Make sure to check out the Schedulers [guide](../../using-diffusers/schedulers.md) to learn how to explore the tradeoff between scheduler speed and quality, and see the [reuse components across pipelines](../../using-diffusers/loading.md#reuse-a-pipeline) section to learn how to efficiently load the same components into multiple pipelines.

This pipeline was contributed by [SHYuanBest](https://github.com/SHYuanBest). The original codebase can be found [here](https://github.com/PKU-YuanGroup/ConsisID). The original weights can be found under [hf.co/BestWishYsh](https://huggingface.co/BestWishYsh).

There are two official ConsisID checkpoints for identity-preserving text-to-video.

| checkpoints | recommended inference dtype |
|:---:|:---:|
| [`BestWishYsh/ConsisID-preview`](https://huggingface.co/BestWishYsh/ConsisID-preview) | torch.bfloat16 |
| [`BestWishYsh/ConsisID-1.5`](https://huggingface.co/BestWishYsh/ConsisID-preview) | torch.bfloat16 |

### Memory optimization

ConsisID requires about 44 GB of GPU memory to decode 49 frames (6 seconds of video at 8 FPS) with output resolution 720x480 (W x H), which makes it not possible to run on consumer GPUs or free-tier T4 Colab. The following memory optimizations could be used to reduce the memory footprint. For replication, you can refer to [this](https://gist.github.com/SHYuanBest/bc4207c36f454f9e969adbb50eaf8258) script.

| Feature (overlay the previous) | Max Memory Allocated | Max Memory Reserved |
| :----------------------------- | :------------------- | :------------------ |
| -                              | 37 GB                | 44 GB               |
| enable_model_cpu_offload       | 22 GB                | 25 GB               |
| enable_sequential_cpu_offload  | 16 GB                | 22 GB               |
| vae.enable_slicing             | 16 GB                | 22 GB               |
| vae.enable_tiling              | 5 GB                 | 7 GB                |

## Load Model Checkpoints

Model weights may be stored in separate subfolders on the Hub or locally, in which case, you should use the [`~DiffusionPipeline.from_pretrained`] method.

```python
# !pip install consisid_eva_clip insightface facexlib
import torch
from diffusers import ConsisIDPipeline
from diffusers.pipelines.consisid.consisid_utils import prepare_face_models, process_face_embeddings_infer
from huggingface_hub import snapshot_download

# Download ckpts
snapshot_download(repo_id="BestWishYsh/ConsisID-preview", local_dir="BestWishYsh/ConsisID-preview")

# Load face helper model to preprocess input face image
face_helper_1, face_helper_2, face_clip_model, face_main_model, eva_transform_mean, eva_transform_std = prepare_face_models("BestWishYsh/ConsisID-preview", device="cuda", dtype=torch.bfloat16)

# Load consisid base model
pipe = ConsisIDPipeline.from_pretrained("BestWishYsh/ConsisID-preview", torch_dtype=torch.bfloat16)
pipe.to("cuda")
```

## Identity-Preserving Text-to-Video

For identity-preserving text-to-video, pass a text prompt and an image contain clear face (e.g., preferably half-body or full-body). By default, ConsisID generates a 720x480 video for the best results.

```python
from diffusers.utils import export_to_video

prompt = "The video captures a boy walking along a city street, filmed in black and white on a classic 35mm camera. His expression is thoughtful, his brow slightly furrowed as if he's lost in contemplation. The film grain adds a textured, timeless quality to the image, evoking a sense of nostalgia. Around him, the cityscape is filled with vintage buildings, cobblestone sidewalks, and softly blurred figures passing by, their outlines faint and indistinct. Streetlights cast a gentle glow, while shadows play across the boy's path, adding depth to the scene. The lighting highlights the boy's subtle smile, hinting at a fleeting moment of curiosity. The overall cinematic atmosphere, complete with classic film still aesthetics and dramatic contrasts, gives the scene an evocative and introspective feel."
image = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/consisid/consisid_input.png?download=true"

id_cond, id_vit_hidden, image, face_kps = process_face_embeddings_infer(face_helper_1, face_clip_model, face_helper_2, eva_transform_mean, eva_transform_std, face_main_model, "cuda", torch.bfloat16, image, is_align_face=True)

video = pipe(image=image, prompt=prompt, num_inference_steps=50, guidance_scale=6.0, use_dynamic_cfg=False, id_vit_hidden=id_vit_hidden, id_cond=id_cond, kps_cond=face_kps, generator=torch.Generator("cuda").manual_seed(42))
export_to_video(video.frames[0], "output.mp4", fps=8)
```
<table>
  <tr>
    <th style="text-align: center;">Face Image</th>
    <th style="text-align: center;">Video</th>
    <th style="text-align: center;">Description</th>
  </tr>
  <tr>
    <td><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/consisid/consisid_image_0.png?download=true" style="height: auto; width: 600px;"></td>
    <td><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/consisid/consisid_output_0.gif?download=true" style="height: auto; width: 2000px;"></td>
    <td>The video, in a beautifully crafted animated style, features a confident woman riding a horse through a lush forest clearing. Her expression is focused yet serene as she adjusts her wide-brimmed hat with a practiced hand. She wears a flowy bohemian dress, which moves gracefully with the rhythm of the horse, the fabric flowing fluidly in the animated motion. The dappled sunlight filters through the trees, casting soft, painterly patterns on the forest floor. Her posture is poised, showing both control and elegance as she guides the horse with ease. The animation's gentle, fluid style adds a dreamlike quality to the scene, with the woman’s calm demeanor and the peaceful surroundings evoking a sense of freedom and harmony.</td>
  </tr>
  <tr>
    <td><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/consisid/consisid_image_1.png?download=true" style="height: auto; width: 600px;"></td>
    <td><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/consisid/consisid_output_1.gif?download=true" style="height: auto; width: 2000px;"></td>
    <td>The video, in a captivating animated style, shows a woman standing in the center of a snowy forest, her eyes narrowed in concentration as she extends her hand forward. She is dressed in a deep blue cloak, her breath visible in the cold air, which is rendered with soft, ethereal strokes. A faint smile plays on her lips as she summons a wisp of ice magic, watching with focus as the surrounding trees and ground begin to shimmer and freeze, covered in delicate ice crystals. The animation’s fluid motion brings the magic to life, with the frost spreading outward in intricate, sparkling patterns. The environment is painted with soft, watercolor-like hues, enhancing the magical, dreamlike atmosphere. The overall mood is serene yet powerful, with the quiet winter air amplifying the delicate beauty of the frozen scene.</td>
  </tr>
  <tr>
    <td><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/consisid/consisid_image_2.png?download=true" style="height: auto; width: 600px;"></td>
    <td><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/consisid/consisid_output_2.gif?download=true" style="height: auto; width: 2000px;"></td>
    <td>The animation features a whimsical portrait of a balloon seller standing in a gentle breeze, captured with soft, hazy brushstrokes that evoke the feel of a serene spring day. His face is framed by a gentle smile, his eyes squinting slightly against the sun, while a few wisps of hair flutter in the wind. He is dressed in a light, pastel-colored shirt, and the balloons around him sway with the wind, adding a sense of playfulness to the scene. The background blurs softly, with hints of a vibrant market or park, enhancing the light-hearted, yet tender mood of the moment.</td>
  </tr>
  <tr>
    <td><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/consisid/consisid_image_3.png?download=true" style="height: auto; width: 600px;"></td>
    <td><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/consisid/consisid_output_3.gif?download=true" style="height: auto; width: 2000px;"></td>
    <td>The video captures a boy walking along a city street, filmed in black and white on a classic 35mm camera. His expression is thoughtful, his brow slightly furrowed as if he's lost in contemplation. The film grain adds a textured, timeless quality to the image, evoking a sense of nostalgia. Around him, the cityscape is filled with vintage buildings, cobblestone sidewalks, and softly blurred figures passing by, their outlines faint and indistinct. Streetlights cast a gentle glow, while shadows play across the boy's path, adding depth to the scene. The lighting highlights the boy's subtle smile, hinting at a fleeting moment of curiosity. The overall cinematic atmosphere, complete with classic film still aesthetics and dramatic contrasts, gives the scene an evocative and introspective feel.</td>
  </tr>
  <tr>
    <td><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/consisid/consisid_image_4.png?download=true" style="height: auto; width: 600px;"></td>
    <td><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/consisid/consisid_output_4.gif?download=true" style="height: auto; width: 2000px;"></td>
    <td>The video features a baby wearing a bright superhero cape, standing confidently with arms raised in a powerful pose. The baby has a determined look on their face, with eyes wide and lips pursed in concentration, as if ready to take on a challenge. The setting appears playful, with colorful toys scattered around and a soft rug underfoot, while sunlight streams through a nearby window, highlighting the fluttering cape and adding to the impression of heroism. The overall atmosphere is lighthearted and fun, with the baby's expressions capturing a mix of innocence and an adorable attempt at bravery, as if truly ready to save the day.</td>
  </tr>
</table>

## Resources

Learn more about ConsisID with the following resources.
- A [video](https://www.youtube.com/watch?v=PhlgC-bI5SQ) demonstrating ConsisID's main features.
- The research paper, [Identity-Preserving Text-to-Video Generation by Frequency Decomposition](https://hf.co/papers/2411.17440) for more details.

## ConsisIDPipeline

[[autodoc]] ConsisIDPipeline

  - all
  - __call__

## ConsisIDPipelineOutput

[[autodoc]] pipelines.consisid.pipeline_output.ConsisIDPipelineOutput
