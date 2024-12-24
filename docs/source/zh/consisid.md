<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->
# ConsisID

[ConsisID](https://github.com/PKU-YuanGroup/ConsisID)是一种身份保持的文本到视频生成模型，其通过频率分解在生成的视频中保持面部一致性。它具有以下特点：

- 基于频率分解：将人物ID特征解耦为高频和低频部分，从频域的角度分析DIT架构的特性，并且基于此特性设计合理的控制信息注入方式。

- 一致性训练策略：我们提出粗到细训练策略、动态掩码损失、动态跨脸损失，进一步提高了模型的泛化能力和身份保持效果。


- 推理无需微调：之前的方法在推理前，需要对输入id进行case-by-case微调，时间和算力开销较大，而我们的方法是tuning-free的。


本指南将指导您使用 ConsisID 生成身份保持的视频。

## Load Model Checkpoints
模型权重可以存储在Hub上或本地的单独子文件夹中，在这种情况下，您应该使用 [`~DiffusionPipeline.from_pretrained`] 方法。


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
对于身份保持的文本到视频生成，需要输入文本提示和包含清晰面部（例如，最好是半身或全身）的图像。默认情况下，ConsisID 会生成 720x480 的视频以获得最佳效果。

```python
from diffusers.utils import export_to_video

prompt = "The video captures a boy walking along a city street, filmed in black and white on a classic 35mm camera. His expression is thoughtful, his brow slightly furrowed as if he's lost in contemplation. The film grain adds a textured, timeless quality to the image, evoking a sense of nostalgia. Around him, the cityscape is filled with vintage buildings, cobblestone sidewalks, and softly blurred figures passing by, their outlines faint and indistinct. Streetlights cast a gentle glow, while shadows play across the boy's path, adding depth to the scene. The lighting highlights the boy's subtle smile, hinting at a fleeting moment of curiosity. The overall cinematic atmosphere, complete with classic film still aesthetics and dramatic contrasts, gives the scene an evocative and introspective feel."
image = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/refs%2Fpr%2F406/diffusers/consisid/consisid_input.png?download=true"

id_cond, id_vit_hidden, image, face_kps = process_face_embeddings_infer(face_helper_1, face_clip_model, face_helper_2, eva_transform_mean, eva_transform_std, face_main_model, "cuda", torch.bfloat16, image, is_align_face=True)

video = pipe(image=image, prompt=prompt, num_inference_steps=50, guidance_scale=6.0, use_dynamic_cfg=False, id_vit_hidden=id_vit_hidden, id_cond=id_cond, kps_cond=face_kps, generator=torch.Generator("cuda").manual_seed(42))
export_to_video(video.frames[0], "output.mp4", fps=8)
```
<table>
  <tr>
    <th style="text-align: center;">Face Image</th>
    <th style="text-align: center;">Video</th>
    <th style="text-align: center;">Description</th
  </tr>
  <tr>
    <td><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/refs%2Fpr%2F406/diffusers/consisid/consisid_image_0.png?download=true" style="height: auto; width: 600px;"></td>
    <td><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/refs%2Fpr%2F406/diffusers/consisid/consisid_output_0.gif?download=true" style="height: auto; width: 2000px;"></td>
    <td>The video, in a beautifully crafted animated style, features a confident woman riding a horse through a lush forest clearing. Her expression is focused yet serene as she adjusts her wide-brimmed hat with a practiced hand. She wears a flowy bohemian dress ......</td>
  </tr>
  <tr>
    <td><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/refs%2Fpr%2F406/diffusers/consisid/consisid_image_1.png?download=true" style="height: auto; width: 600px;"></td>
    <td><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/refs%2Fpr%2F406/diffusers/consisid/consisid_output_1.gif?download=true" style="height: auto; width: 2000px;"></td>
    <td>The video, in a captivating animated style, shows a woman standing in the center of a snowy forest, her eyes narrowed in concentration as she extends her hand forward. She is dressed in a deep blue cloak, her breath visible in the cold air ......</td>
  </tr>
  <tr>
    <td><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/refs%2Fpr%2F406/diffusers/consisid/consisid_image_2.png?download=true" style="height: auto; width: 600px;"></td>
    <td><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/refs%2Fpr%2F406/diffusers/consisid/consisid_output_2.gif?download=true" style="height: auto; width: 2000px;"></td>
    <td>The animation features a whimsical portrait of a balloon seller standing in a gentle breeze, captured with soft, hazy brushstrokes that evoke the feel of a serene spring day. His face is framed by a gentle smile, his eyes ......</td>
  </tr>
  <tr>
    <td><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/refs%2Fpr%2F406/diffusers/consisid/consisid_image_3.png?download=true" style="height: auto; width: 600px;"></td>
    <td><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/refs%2Fpr%2F406/diffusers/consisid/consisid_output_3.gif?download=true" style="height: auto; width: 2000px;"></td>
    <td>The video captures a boy walking along a city street, filmed in black and white on a classic 35mm camera. His expression is thoughtful, his brow slightly furrowed as if he's lost in contemplation. The film grain adds a textured ......</td>
  </tr>
  <tr>
    <td><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/refs%2Fpr%2F406/diffusers/consisid/consisid_image_4.png?download=true" style="height: auto; width: 600px;"></td>
    <td><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/refs%2Fpr%2F406/diffusers/consisid/consisid_output_4.gif?download=true" style="height: auto; width: 2000px;"></td>
    <td>The video features a baby wearing a bright superhero cape, standing confidently with arms raised in a powerful pose. The baby has a determined look on their face, with eyes wide and lips pursed in concentration, as if ready to take on a challenge ......</td>
  </tr>
</table>

## Citation

通过以下资源了解有关 ConsisID 的更多信息：

- 一段 [视频](https://www.youtube.com/watch?v=PhlgC-bI5SQ) 演示了 ConsisID 的主要功能；
- 有关更多详细信息，请参阅研究论文 [Identity-Preserving Text-to-Video Generation by Frequency Decomposition](https://hf.co/papers/2411.17440)。
