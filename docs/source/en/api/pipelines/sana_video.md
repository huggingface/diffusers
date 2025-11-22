<!-- Copyright 2025 The SANA-Video Authors and HuggingFace Team. All rights reserved.
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

# Sana-Video

<div class="flex flex-wrap space-x-1">
  <img alt="LoRA" src="https://img.shields.io/badge/LoRA-d8b4fe?style=flat"/>
  <img alt="MPS" src="https://img.shields.io/badge/MPS-000000?style=flat&logo=apple&logoColor=white%22">
</div>

[SANA-Video: Efficient Video Generation with Block Linear Diffusion Transformer](https://huggingface.co/papers/2509.24695) from NVIDIA and MIT HAN Lab, by Junsong Chen, Yuyang Zhao, Jincheng Yu, Ruihang Chu, Junyu Chen, Shuai Yang, Xianbang Wang, Yicheng Pan, Daquan Zhou, Huan Ling, Haozhe Liu, Hongwei Yi, Hao Zhang, Muyang Li, Yukang Chen, Han Cai, Sanja Fidler, Ping Luo, Song Han, Enze Xie.

The abstract from the paper is:

*We introduce SANA-Video, a small diffusion model that can efficiently generate videos up to 720x1280 resolution and minute-length duration. SANA-Video synthesizes high-resolution, high-quality and long videos with strong text-video alignment at a remarkably fast speed, deployable on RTX 5090 GPU. Two core designs ensure our efficient, effective and long video generation: (1) Linear DiT: We leverage linear attention as the core operation, which is more efficient than vanilla attention given the large number of tokens processed in video generation. (2) Constant-Memory KV cache for Block Linear Attention: we design block-wise autoregressive approach for long video generation by employing a constant-memory state, derived from the cumulative properties of linear attention. This KV cache provides the Linear DiT with global context at a fixed memory cost, eliminating the need for a traditional KV cache and enabling efficient, minute-long video generation. In addition, we explore effective data filters and model training strategies, narrowing the training cost to 12 days on 64 H100 GPUs, which is only 1% of the cost of MovieGen. Given its low cost, SANA-Video achieves competitive performance compared to modern state-of-the-art small diffusion models (e.g., Wan 2.1-1.3B and SkyReel-V2-1.3B) while being 16x faster in measured latency. Moreover, SANA-Video can be deployed on RTX 5090 GPUs with NVFP4 precision, accelerating the inference speed of generating a 5-second 720p video from 71s to 29s (2.4x speedup). In summary, SANA-Video enables low-cost, high-quality video generation. [this https URL](https://github.com/NVlabs/SANA).*

This pipeline was contributed by SANA Team. The original codebase can be found [here](https://github.com/NVlabs/Sana). The original weights can be found under [hf.co/Efficient-Large-Model](https://hf.co/collections/Efficient-Large-Model/sana-video).

Available models:

| Model | Recommended dtype |
|:-----:|:-----------------:|
| [`Efficient-Large-Model/SANA-Video_2B_480p_diffusers`](https://huggingface.co/Efficient-Large-Model/ANA-Video_2B_480p_diffusers) | `torch.bfloat16` |

Refer to [this](https://huggingface.co/collections/Efficient-Large-Model/sana-video) collection for more information.

Note: The recommended dtype mentioned is for the transformer weights. The text encoder and VAE weights must stay in `torch.bfloat16` or `torch.float32` for the model to work correctly. Please refer to the inference example below to see how to load the model with the recommended dtype. 


## Generation Pipelines

<hfoptions id="generation pipelines">`
<hfoption id="Text-to-Video">

The example below demonstrates how to use the text-to-video pipeline to generate a video using a text descriptio and a starting frame.

```python
model_id = 
pipe = SanaVideoPipeline.from_pretrained("Efficient-Large-Model/SANA-Video_2B_480p_diffusers", torch_dtype=torch.bfloat16)
pipe.text_encoder.to(torch.bfloat16)
pipe.vae.to(torch.float32)
pipe.to("cuda")

prompt = "A cat and a dog baking a cake together in a kitchen. The cat is carefully measuring flour, while the dog is stirring the batter with a wooden spoon. The kitchen is cozy, with sunlight streaming through the window."
negative_prompt = "A chaotic sequence with misshapen, deformed limbs in heavy motion blur, sudden disappearance, jump cuts, jerky movements, rapid shot changes, frames out of sync, inconsistent character shapes, temporal artifacts, jitter, and ghosting effects, creating a disorienting visual experience."
motion_scale = 30
motion_prompt = f" motion score: {motion_scale}."
prompt = prompt + motion_prompt

video = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=480,
    width=832,
    frames=81,
    guidance_scale=6,
    num_inference_steps=50,
    generator=torch.Generator(device="cuda").manual_seed(0),
).frames[0]

export_to_video(video, "sana_video.mp4", fps=16)
```

</hfoption>
<hfoption id="Image-to-Video">

The example below demonstrates how to use the image-to-video pipeline to generate a video using a text descriptio and a starting frame.

```python
model_id = "Efficient-Large-Model/SANA-Video_2B_480p_diffusers"
pipe = SanaImageToVideoPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
)
pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config, flow_shift=8.0)
pipe.vae.to(torch.float32)
pipe.text_encoder.to(torch.bfloat16)
pipe.to("cuda")

image = load_image("https://raw.githubusercontent.com/NVlabs/Sana/refs/heads/main/asset/samples/i2v-1.png")
prompt = "A woman stands against a stunning sunset backdrop, her long, wavy brown hair gently blowing in the breeze. She wears a sleeveless, light-colored blouse with a deep V-neckline, which accentuates her graceful posture. The warm hues of the setting sun cast a golden glow across her face and hair, creating a serene and ethereal atmosphere. The background features a blurred landscape with soft, rolling hills and scattered clouds, adding depth to the scene. The camera remains steady, capturing the tranquil moment from a medium close-up angle."
negative_prompt = "A chaotic sequence with misshapen, deformed limbs in heavy motion blur, sudden disappearance, jump cuts, jerky movements, rapid shot changes, frames out of sync, inconsistent character shapes, temporal artifacts, jitter, and ghosting effects, creating a disorienting visual experience."
motion_scale = 30
motion_prompt = f" motion score: {motion_scale}."
prompt = prompt + motion_prompt

motion_scale = 30.0

video = pipe(
    image=image,
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=480,
    width=832,
    frames=81,
    guidance_scale=6,
    num_inference_steps=50,
    generator=torch.Generator(device="cuda").manual_seed(0),
).frames[0]

export_to_video(video, "sana-i2v.mp4", fps=16)
```

</hfoption>
</hfoptions>


## Quantization

Quantization helps reduce the memory requirements of very large models by storing model weights in a lower precision data type. However, quantization may have varying impact on video quality depending on the video model.

Refer to the [Quantization](../../quantization/overview) overview to learn more about supported quantization backends and selecting a quantization backend that supports your use case. The example below demonstrates how to load a quantized [`SanaVideoPipeline`] for inference with bitsandbytes.

```py
import torch
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig, SanaVideoTransformer3DModel, SanaVideoPipeline
from transformers import BitsAndBytesConfig as BitsAndBytesConfig, AutoModel

quant_config = BitsAndBytesConfig(load_in_8bit=True)
text_encoder_8bit = AutoModel.from_pretrained(
    "Efficient-Large-Model/SANA-Video_2B_480p_diffusers",
    subfolder="text_encoder",
    quantization_config=quant_config,
    torch_dtype=torch.float16,
)

quant_config = DiffusersBitsAndBytesConfig(load_in_8bit=True)
transformer_8bit = SanaVideoTransformer3DModel.from_pretrained(
    "Efficient-Large-Model/SANA-Video_2B_480p_diffusers",
    subfolder="transformer",
    quantization_config=quant_config,
    torch_dtype=torch.float16,
)

pipeline = SanaVideoPipeline.from_pretrained(
    "Efficient-Large-Model/SANA-Video_2B_480p_diffusers",
    text_encoder=text_encoder_8bit,
    transformer=transformer_8bit,
    torch_dtype=torch.float16,
    device_map="balanced",
)

model_score = 30
prompt = "Evening, backlight, side lighting, soft light, high contrast, mid-shot, centered composition, clean solo shot, warm color. A young Caucasian man stands in a forest, golden light glimmers on his hair as sunlight filters through the leaves. He wears a light shirt, wind gently blowing his hair and collar, light dances across his face with his movements. The background is blurred, with dappled light and soft tree shadows in the distance. The camera focuses on his lifted gaze, clear and emotional."
negative_prompt = "A chaotic sequence with misshapen, deformed limbs in heavy motion blur, sudden disappearance, jump cuts, jerky movements, rapid shot changes, frames out of sync, inconsistent character shapes, temporal artifacts, jitter, and ghosting effects, creating a disorienting visual experience."
motion_prompt = f" motion score: {model_score}."
prompt = prompt + motion_prompt

output = pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=480,
    width=832,
    num_frames=81,
    guidance_scale=6.0,
    num_inference_steps=50
).frames[0]
export_to_video(output, "sana-video-output.mp4", fps=16)
```

## SanaVideoPipeline

[[autodoc]] SanaVideoPipeline
  - all
  - __call__


## SanaImageToVideoPipeline

[[autodoc]] SanaImageToVideoPipeline
  - all
  - __call__


## SanaVideoPipelineOutput

[[autodoc]] pipelines.sana_video.pipeline_sana_video.SanaVideoPipelineOutput
