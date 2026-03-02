<!--Copyright 2025 The HuggingFace Team and Kandinsky Lab Team. All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Kandinsky 5.0 Image

[Kandinsky 5.0](https://arxiv.org/abs/2511.14993) is a family of diffusion models for Video & Image generation. 

Kandinsky 5.0 Image Lite is a lightweight image generation model (6B parameters).

The model introduces several key innovations:
- **Latent diffusion pipeline** with **Flow Matching** for improved training stability
- **Diffusion Transformer (DiT)** as the main generative backbone with cross-attention to text embeddings
- Dual text encoding using **Qwen2.5-VL** and **CLIP** for comprehensive text understanding
- **Flux VAE** for efficient image encoding and decoding

The original codebase can be found at [kandinskylab/Kandinsky-5](https://github.com/kandinskylab/Kandinsky-5).

> [!TIP]
> Check out the [Kandinsky Lab](https://huggingface.co/kandinskylab) organization on the Hub for the official model checkpoints for text-to-video generation, including pretrained, SFT, no-CFG, and distilled variants.


## Available Models

Kandinsky 5.0 Image Lite:

| model_id | Description | Use Cases |
|------------|-------------|-----------|
| [**kandinskylab/Kandinsky-5.0-T2I-Lite-sft-Diffusers**](https://huggingface.co/kandinskylab/Kandinsky-5.0-T2I-Lite-sft-Diffusers) | 6B image Supervised Fine-Tuned model | Highest generation quality |
| [**kandinskylab/Kandinsky-5.0-I2I-Lite-sft-Diffusers**](https://huggingface.co/kandinskylab/Kandinsky-5.0-I2I-Lite-sft-Diffusers) | 6B image editing Supervised Fine-Tuned model | Highest generation quality |
| [**kandinskylab/Kandinsky-5.0-T2I-Lite-pretrain-Diffusers**](https://huggingface.co/kandinskylab/Kandinsky-5.0-T2I-Lite-pretrain-Diffusers) | 6B image Base pretrained model | Research and fine-tuning |
| [**kandinskylab/Kandinsky-5.0-I2I-Lite-pretrain-Diffusers**](https://huggingface.co/kandinskylab/Kandinsky-5.0-I2I-Lite-pretrain-Diffusers) | 6B image editing Base pretrained model | Research and fine-tuning |

## Usage Examples

### Basic Text-to-Image Generation

```python
import torch
from diffusers import Kandinsky5T2IPipeline

# Load the pipeline
model_id = "kandinskylab/Kandinsky-5.0-T2I-Lite-sft-Diffusers"
pipe = Kandinsky5T2IPipeline.from_pretrained(model_id)
_ = pipe.to(device='cuda',dtype=torch.bfloat16)

# Generate image
prompt = "A fluffy, expressive cat wearing a bright red hat with a soft, slightly textured fabric. The hat should look cozy and well-fitted on the cat’s head. On the front of the hat, add clean, bold white text that reads “SWEET”, clearly visible and neatly centered. Ensure the overall lighting highlights the hat’s color and the cat’s fur details."

output = pipe(
    prompt=prompt,
    negative_prompt="",
    height=1024,
    width=1024,
    num_inference_steps=50,
    guidance_scale=3.5,
).image[0]
```

### Basic Image-to-Image Generation

```python
import torch
from diffusers import Kandinsky5I2IPipeline
from diffusers.utils import load_image 
# Load the pipeline
model_id = "kandinskylab/Kandinsky-5.0-I2I-Lite-sft-Diffusers"
pipe = Kandinsky5I2IPipeline.from_pretrained(model_id)

_ = pipe.to(device='cuda',dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()                                               # <--- Enable CPU offloading for single GPU inference

# Edit the input image
image = load_image(
    "https://huggingface.co/kandinsky-community/kandinsky-3/resolve/main/assets/title.jpg?download=true"
)

prompt = "Change the background from a winter night scene to a bright summer day. Place the character on a sandy beach with clear blue sky, soft sunlight, and gentle waves in the distance. Replace the winter clothing with a light short-sleeved T-shirt (in soft pastel colors) and casual shorts. Ensure the character’s fur reflects warm daylight instead of cold winter tones. Add small beach details such as seashells, footprints in the sand, and a few scattered beach toys nearby. Keep the oranges in the scene, but place them naturally on the sand."
negative_prompt = ""

output = pipe(
    image=image,
    prompt=prompt,
    negative_prompt=negative_prompt,
    guidance_scale=3.5,
).image[0]
```


## Kandinsky5T2IPipeline

[[autodoc]] Kandinsky5T2IPipeline
    - all
    - __call__

## Kandinsky5I2IPipeline

[[autodoc]] Kandinsky5I2IPipeline
    - all
    - __call__


## Citation
```bibtex
@misc{kandinsky2025,
    author = {Alexander Belykh and Alexander Varlamov and Alexey Letunovskiy and Anastasia Aliaskina and Anastasia Maltseva and Anastasiia Kargapoltseva and Andrey Shutkin and Anna Averchenkova and Anna Dmitrienko and Bulat Akhmatov and Denis Dimitrov and Denis Koposov and Denis Parkhomenko and Dmitrii and Ilya Vasiliev and Ivan Kirillov and Julia Agafonova and Kirill Chernyshev and Kormilitsyn Semen and Lev Novitskiy and Maria Kovaleva and Mikhail Mamaev and Mikhailov and Nikita Kiselev and Nikita Osterov and Nikolai Gerasimenko and Nikolai Vaulin and Olga Kim and Olga Vdovchenko and Polina Gavrilova and Polina Mikhailova and Tatiana Nikulina and Viacheslav Vasilev and Vladimir Arkhipkin and Vladimir Korviakov and Vladimir Polovnikov and Yury Kolabushin},
    title = {Kandinsky 5.0: A family of diffusion models for Video & Image generation},
    howpublished = {\url{https://github.com/kandinskylab/Kandinsky-5}},
    year = 2025
}
```
