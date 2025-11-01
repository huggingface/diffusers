<!--Copyright 2025 The HuggingFace Team. All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Kandinsky 5.0

Kandinsky 5.0 is created by the Kandinsky team: Alexey Letunovskiy, Maria Kovaleva, Ivan Kirillov, Lev Novitskiy, Denis Koposov, Dmitrii Mikhailov, Anna Averchenkova, Andrey Shutkin, Julia Agafonova, Olga Kim, Anastasiia Kargapoltseva, Nikita Kiselev, Anna Dmitrienko,  Anastasia Maltseva, Kirill Chernyshev, Ilia Vasiliev, Viacheslav Vasilev, Vladimir Polovnikov, Yury Kolabushin, Alexander Belykh, Mikhail Mamaev, Anastasia Aliaskina, Tatiana Nikulina, Polina Gavrilova, Vladimir Arkhipkin, Vladimir Korviakov, Nikolai Gerasimenko, Denis Parkhomenko, Denis Dimitrov


Kandinsky 5.0 is a family of diffusion models for Video & Image generation. Kandinsky 5.0 T2V Lite is a lightweight video generation model (2B parameters) that ranks #1 among open-source models in its class. It outperforms larger models and offers the best understanding of Russian concepts in the open-source ecosystem.

The model introduces several key innovations:
- **Latent diffusion pipeline** with **Flow Matching** for improved training stability
- **Diffusion Transformer (DiT)** as the main generative backbone with cross-attention to text embeddings
- Dual text encoding using **Qwen2.5-VL** and **CLIP** for comprehensive text understanding
- **HunyuanVideo 3D VAE** for efficient video encoding and decoding
- **Sparse attention mechanisms** (NABLA) for efficient long-sequence processing

The original codebase can be found at [ai-forever/Kandinsky-5](https://github.com/ai-forever/Kandinsky-5).

> [!TIP]
> Check out the [AI Forever](https://huggingface.co/ai-forever) organization on the Hub for the official model checkpoints for text-to-video generation, including pretrained, SFT, no-CFG, and distilled variants.

## Available Models

Kandinsky 5.0 T2V Lite comes in several variants optimized for different use cases:

| model_id | Description | Use Cases |
|------------|-------------|-----------|
| **ai-forever/Kandinsky-5.0-T2V-Lite-sft-5s-Diffusers** | 5 second Supervised Fine-Tuned model | Highest generation quality |
| **ai-forever/Kandinsky-5.0-T2V-Lite-sft-10s-Diffusers** | 10 second Supervised Fine-Tuned model | Highest generation quality |
| **ai-forever/Kandinsky-5.0-T2V-Lite-nocfg-5s-Diffusers** | 5 second Classifier-Free Guidance distilled | 2× faster inference |
| **ai-forever/Kandinsky-5.0-T2V-Lite-nocfg-10s-Diffusers** | 10 second Classifier-Free Guidance distilled | 2× faster inference |
| **ai-forever/Kandinsky-5.0-T2V-Lite-distilled16steps-5s-Diffusers** | 5 second Diffusion distilled to 16 steps | 6× faster inference, minimal quality loss |
| **ai-forever/Kandinsky-5.0-T2V-Lite-distilled16steps-10s-Diffusers** | 10 second Diffusion distilled to 16 steps | 6× faster inference, minimal quality loss |
| **ai-forever/Kandinsky-5.0-T2V-Lite-pretrain-5s-Diffusers** | 5 second Base pretrained model | Research and fine-tuning |
| **ai-forever/Kandinsky-5.0-T2V-Lite-pretrain-10s-Diffusers** | 10 second Base pretrained model | Research and fine-tuning |

All models are available in 5-second and 10-second video generation versions.

## Kandinsky5T2VPipeline

[[autodoc]] Kandinsky5T2VPipeline
    - all
    - __call__

## Usage Examples

### Basic Text-to-Video Generation

```python
import torch
from diffusers import Kandinsky5T2VPipeline
from diffusers.utils import export_to_video

# Load the pipeline
model_id = "ai-forever/Kandinsky-5.0-T2V-Lite-sft-5s-Diffusers"
pipe = Kandinsky5T2VPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
pipe = pipe.to("cuda")

# Generate video
prompt = "A cat and a dog baking a cake together in a kitchen."
negative_prompt = "Static, 2D cartoon, cartoon, 2d animation, paintings, images, worst quality, low quality, ugly, deformed, walking backwards"

output = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=512,
    width=768,
    num_frames=121,  # ~5 seconds at 24fps
    num_inference_steps=50,
    guidance_scale=5.0,
).frames[0]

export_to_video(output, "output.mp4", fps=24, quality=9)
```

### 10 second Models
**⚠️ Warning!** all 10 second models should be used with Flex attention and max-autotune-no-cudagraphs compilation:

```python
pipe = Kandinsky5T2VPipeline.from_pretrained(
    "ai-forever/Kandinsky-5.0-T2V-Lite-sft-10s-Diffusers", 
    torch_dtype=torch.bfloat16
)
pipe = pipe.to("cuda")

pipe.transformer.set_attention_backend(
    "flex"
)                                       # <--- Set attention backend to Flex
pipe.transformer.compile(
    mode="max-autotune-no-cudagraphs", 
    dynamic=True
)                                       # <--- Compile with max-autotune-no-cudagraphs

prompt = "A cat and a dog baking a cake together in a kitchen."
negative_prompt = "Static, 2D cartoon, cartoon, 2d animation, paintings, images, worst quality, low quality, ugly, deformed, walking backwards"

output = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=512,
    width=768,
    num_frames=241,
    num_inference_steps=50,
    guidance_scale=5.0,
).frames[0]

export_to_video(output, "output.mp4", fps=24, quality=9)
```

### Diffusion Distilled model
**⚠️ Warning!** all nocfg and diffusion distilled models should be inferred without CFG (```guidance_scale=1.0```):

```python
model_id = "ai-forever/Kandinsky-5.0-T2V-Lite-distilled16steps-5s-Diffusers"
pipe = Kandinsky5T2VPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
pipe = pipe.to("cuda")

output = pipe(
    prompt="A beautiful sunset over mountains",
    num_inference_steps=16,  # <--- Model is distilled in 16 steps
    guidance_scale=1.0,      # <--- no CFG
).frames[0]

export_to_video(output, "output.mp4", fps=24, quality=9)
```


## Citation
```bibtex
@misc{kandinsky2025,
    author = {Alexey Letunovskiy and Maria Kovaleva and Ivan Kirillov and Lev Novitskiy and Denis Koposov and
              Dmitrii Mikhailov and Anna Averchenkova and Andrey Shutkin and Julia Agafonova and Olga Kim and
              Anastasiia Kargapoltseva and Nikita Kiselev and Vladimir Arkhipkin and Vladimir Korviakov and
              Nikolai Gerasimenko and Denis Parkhomenko and Anna Dmitrienko and Anastasia Maltseva and
              Kirill Chernyshev and Ilia Vasiliev and Viacheslav Vasilev and Vladimir Polovnikov and
              Yury Kolabushin and Alexander Belykh and Mikhail Mamaev and Anastasia Aliaskina and
              Tatiana Nikulina and Polina Gavrilova and Denis Dimitrov},
    title = {Kandinsky 5.0: A family of diffusion models for Video & Image generation},
    howpublished = {\url{https://github.com/ai-forever/Kandinsky-5}},
    year = 2025
}
```
