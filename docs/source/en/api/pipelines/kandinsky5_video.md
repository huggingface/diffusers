<!--Copyright 2025 The HuggingFace Team Kandinsky Lab Team. All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Kandinsky 5.0 Video

[Kandinsky 5.0](https://arxiv.org/abs/2511.14993) is a family of diffusion models for Video & Image generation.

Kandinsky 5.0 Lite line-up of lightweight video generation models (2B parameters) that ranks #1 among open-source models in its class. It outperforms larger models and offers the best understanding of Russian concepts in the open-source ecosystem.

Kandinsky 5.0 Pro line-up of large high quality video generation models (19B parameters). It offers high qualty generation in HD and more generation formats like I2V.

The model introduces several key innovations:
- **Latent diffusion pipeline** with **Flow Matching** for improved training stability
- **Diffusion Transformer (DiT)** as the main generative backbone with cross-attention to text embeddings
- Dual text encoding using **Qwen2.5-VL** and **CLIP** for comprehensive text understanding
- **HunyuanVideo 3D VAE** for efficient video encoding and decoding
- **Sparse attention mechanisms** (NABLA) for efficient long-sequence processing

The original codebase can be found at [kandinskylab/Kandinsky-5](https://github.com/kandinskylab/Kandinsky-5).

> [!TIP]
> Check out the [Kandinsky Lab](https://huggingface.co/kandinskylab) organization on the Hub for the official model checkpoints for text-to-video generation, including pretrained, SFT, no-CFG, and distilled variants.

## Available Models

Kandinsky 5.0 T2V Pro:

| model_id | Description | Use Cases |
|------------|-------------|-----------|
| **kandinskylab/Kandinsky-5.0-T2V-Pro-sft-5s-Diffusers** | 5 second Text-to-Video Pro model | High-quality text-to-video generation |
| **kandinskylab/Kandinsky-5.0-I2V-Pro-sft-5s-Diffusers** | 5 second Image-to-Video Pro model | High-quality image-to-video generation |

Kandinsky 5.0 T2V Lite:
| model_id | Description | Use Cases |
|------------|-------------|-----------|
| **kandinskylab/Kandinsky-5.0-T2V-Lite-sft-5s-Diffusers** | 5 second Supervised Fine-Tuned model | Highest generation quality |
| **kandinskylab/Kandinsky-5.0-T2V-Lite-sft-10s-Diffusers** | 10 second Supervised Fine-Tuned model | Highest generation quality |
| **kandinskylab/Kandinsky-5.0-T2V-Lite-nocfg-5s-Diffusers** | 5 second Classifier-Free Guidance distilled | 2× faster inference |
| **kandinskylab/Kandinsky-5.0-T2V-Lite-nocfg-10s-Diffusers** | 10 second Classifier-Free Guidance distilled | 2× faster inference |
| **kandinskylab/Kandinsky-5.0-T2V-Lite-distilled16steps-5s-Diffusers** | 5 second Diffusion distilled to 16 steps | 6× faster inference, minimal quality loss |
| **kandinskylab/Kandinsky-5.0-T2V-Lite-distilled16steps-10s-Diffusers** | 10 second Diffusion distilled to 16 steps | 6× faster inference, minimal quality loss |
| **kandinskylab/Kandinsky-5.0-T2V-Lite-pretrain-5s-Diffusers** | 5 second Base pretrained model | Research and fine-tuning |
| **kandinskylab/Kandinsky-5.0-T2V-Lite-pretrain-10s-Diffusers** | 10 second Base pretrained model | Research and fine-tuning |


## Usage Examples

### Basic Text-to-Video Generation

#### Pro
**⚠️ Warning!** all Pro models should be infered with pipeline.enable_model_cpu_offload()  
```python
import torch
from diffusers import Kandinsky5T2VPipeline
from diffusers.utils import export_to_video

# Load the pipeline
model_id = "kandinskylab/Kandinsky-5.0-T2V-Pro-sft-5s-Diffusers"
pipe = Kandinsky5T2VPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)

pipe = pipe.to("cuda")
pipeline.transformer.set_attention_backend("flex")                            # <--- Set attention bakend to Flex
pipeline.enable_model_cpu_offload()                                           # <--- Enable cpu offloading for single GPU inference
pipeline.transformer.compile(mode="max-autotune-no-cudagraphs", dynamic=True) # <--- Compile with max-autotune-no-cudagraphs

# Generate video
prompt = "A cat and a dog baking a cake together in a kitchen."
negative_prompt = "Static, 2D cartoon, cartoon, 2d animation, paintings, images, worst quality, low quality, ugly, deformed, walking backwards"

output = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=768,
    width=1024,
    num_frames=121,  # ~5 seconds at 24fps
    num_inference_steps=50,
    guidance_scale=5.0,
).frames[0]

export_to_video(output, "output.mp4", fps=24, quality=9)
```

#### Lite
```python
import torch
from diffusers import Kandinsky5T2VPipeline
from diffusers.utils import export_to_video

# Load the pipeline
model_id = "kandinskylab/Kandinsky-5.0-T2V-Lite-sft-5s-Diffusers"
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
    "kandinskylab/Kandinsky-5.0-T2V-Lite-sft-10s-Diffusers", 
    torch_dtype=torch.bfloat16
)
pipe = pipe.to("cuda")

pipe.transformer.set_attention_backend(
    "flex"
)                                       # <--- Set attention bakend to Flex
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
**⚠️ Warning!** all nocfg and diffusion distilled models should be infered wothout CFG (```guidance_scale=1.0```):

```python
model_id = "kandinskylab/Kandinsky-5.0-T2V-Lite-distilled16steps-5s-Diffusers"
pipe = Kandinsky5T2VPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
pipe = pipe.to("cuda")

output = pipe(
    prompt="A beautiful sunset over mountains",
    num_inference_steps=16,  # <--- Model is distilled in 16 steps
    guidance_scale=1.0,      # <--- no CFG
).frames[0]

export_to_video(output, "output.mp4", fps=24, quality=9)
```


### Basic Image-to-Video Generation
**⚠️ Warning!** all Pro models should be infered with pipeline.enable_model_cpu_offload()  
```python
import torch
from diffusers import Kandinsky5T2VPipeline
from diffusers.utils import export_to_video

# Load the pipeline
model_id = "kandinskylab/Kandinsky-5.0-I2V-Pro-sft-5s-Diffusers"
pipe = Kandinsky5T2VPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)

pipe = pipe.to("cuda")
pipeline.transformer.set_attention_backend("flex")                            # <--- Set attention bakend to Flex
pipeline.enable_model_cpu_offload()                                           # <--- Enable cpu offloading for single GPU inference
pipeline.transformer.compile(mode="max-autotune-no-cudagraphs", dynamic=True) # <--- Compile with max-autotune-no-cudagraphs

# Generate video
image = load_image(
    "https://huggingface.co/kandinsky-community/kandinsky-3/resolve/main/assets/title.jpg?download=true"
)
height = 896
width = 896
image = image.resize((width, height))

prompt = "An funny furry creture smiles happily and holds a sign that says 'Kandinsky'"
negative_prompt = ""

output = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=height,
    width=width,
    num_frames=121,  # ~5 seconds at 24fps
    num_inference_steps=50,
    guidance_scale=5.0,
).frames[0]

export_to_video(output, "output.mp4", fps=24, quality=9)
```



## Kandinsky 5.0 Pro Side-by-Side evaluation

<table border="0" style="width: 200; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <img width="200" alt="image" src="https://github.com/user-attachments/assets/73e5ff00-2735-40fd-8f01-767de9181918" />
      </td>
      <td>
         <img width="200" alt="image" src="https://github.com/user-attachments/assets/f449a9e7-74b7-481d-82da-02723e396acd" />
      </td>

  <tr>
      <td>
          Comparison with Veo 3 
      </td>
      <td>
          Comparison with Veo 3 fast
      </td>
  <tr>
      <td>
          <img width="200" alt="image" src="https://github.com/user-attachments/assets/a6902fb6-b5e8-4093-adad-aa4caab79c6d" />
      </td>
      <td>
          <img width="200" alt="image" src="https://github.com/user-attachments/assets/09986015-3d07-4de8-b942-c145039b9b2d" />
      </td>
  <tr>
      <td>
          Comparison with Wan 2.2 A14B Text-to-Video mode
      </td>
      <td>
          Comparison with Wan 2.2 A14B Image-to-Video mode
      </td>

</table>


## Kandinsky 5.0 Lite Side-by-Side evaluation

The evaluation is based on the expanded prompts from the [Movie Gen benchmark](https://github.com/facebookresearch/MovieGenBench), which are available in the expanded_prompt column of the benchmark/moviegen_bench.csv file.

<table border="0" style="width: 400; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <img src="https://github.com/kandinskylab/kandinsky-5/raw/main/assets/sbs/kandinsky_5_video_lite_vs_sora.jpg" width=400 >
      </td>
      <td>
          <img src="https://github.com/kandinskylab/kandinsky-5/raw/main/assets/sbs/kandinsky_5_video_lite_vs_wan_2.1_14B.jpg" width=400 >
      </td>
  <tr>
      <td>
          <img src="https://github.com/kandinskylab/kandinsky-5/raw/main/assets/sbs/kandinsky_5_video_lite_vs_wan_2.2_5B.jpg" width=400 >
      </td>
      <td>
          <img src="https://github.com/kandinskylab/kandinsky-5/raw/main/assets/sbs/kandinsky_5_video_lite_vs_wan_2.2_A14B.jpg" width=400 >
      </td>
  <tr>
      <td>
          <img src="https://github.com/kandinskylab/kandinsky-5/raw/main/assets/sbs/kandinsky_5_video_lite_vs_wan_2.1_1.3B.jpg" width=400 >
      </td>

</table>




## Kandinsky 5.0 Lite Distill Side-by-Side evaluation

<table border="0" style="width: 400; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <img src="https://github.com/kandinskylab/kandinsky-5/raw/main/assets/sbs/kandinsky_5_video_lite_5s_vs_kandinsky_5_video_lite_distill_5s.jpg" width=400 >
      </td>
      <td>
          <img src="https://github.com/kandinskylab/kandinsky-5/raw/main/assets/sbs/kandinsky_5_video_lite_10s_vs_kandinsky_5_video_lite_distill_10s.jpg" width=400 >
      </td>

</table>

## Kandinsky5T2VPipeline

[[autodoc]] Kandinsky5T2VPipeline
    - all
    - __call__

## Kandinsky5I2VPipeline

[[autodoc]] Kandinsky5I2VPipeline
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
