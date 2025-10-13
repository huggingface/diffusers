<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Lumina-T2I

[Lumina-T2I](https://arxiv.org/abs/2405.05945) is a 5B parameter text-to-image diffusion transformer that uses LLaMA-2-7B as its text encoder. It implements a rectified flow approach for efficient, high-quality image generation with support for variable resolutions.

This guide will show you how to use Lumina-T2I for text-to-image generation and various advanced use cases.

> [!TIP]
> Lumina-T2I requires access to the LLaMA-2-7B model. Make sure you have accepted the [LLaMA-2 license](https://huggingface.co/meta-llama/Llama-2-7b-hf) on Hugging Face and have your access token ready.

## Loading the pipeline

Load the [`LuminaT2IPipeline`], specify the model checkpoint, and pass your Hugging Face token for accessing the LLaMA-2 model:

```python
import torch
from diffusers import LuminaT2IPipeline

pipeline = LuminaT2IPipeline.from_pretrained(
    "Alpha-VLLM/Lumina-T2I",
    torch_dtype=torch.bfloat16,
    use_auth_token="your_huggingface_token"
)
pipeline = pipeline.to("cuda")
```

## Text-to-image

For text-to-image, pass a text prompt and the pipeline will generate an image:

```python
prompt = "A majestic lion standing on a cliff overlooking a vast savanna at sunset"
image = pipeline(
    prompt=prompt,
    num_inference_steps=30,
    guidance_scale=4.0,
    height=1024,
    width=1024,
).images[0]

image.save("lion_sunset.png")
```

### Adjusting guidance scale

The `guidance_scale` parameter controls how closely the image follows the text prompt. Higher values make the image more aligned with the prompt but may reduce diversity:

```python
# Lower guidance (more creative, diverse)
image = pipeline(prompt, guidance_scale=2.0).images[0]

# Higher guidance (more literal, focused)
image = pipeline(prompt, guidance_scale=7.0).images[0]
```

**Recommended range**: 3.0 to 5.0

### Number of inference steps

More steps generally produce higher quality images but take longer:

```python
# Fast generation (lower quality)
image = pipeline(prompt, num_inference_steps=20).images[0]

# High quality (slower)
image = pipeline(prompt, num_inference_steps=50).images[0]
```

**Recommended**: 30-40 steps for most use cases

## Variable resolution

Lumina-T2I supports flexible resolutions and aspect ratios:

```python
# Square
image = pipeline(prompt, height=1024, width=1024).images[0]

# Landscape
image = pipeline(prompt, height=512, width=2048).images[0]

# Portrait
image = pipeline(prompt, height=2048, width=512).images[0]

# Wide panorama
image = pipeline(prompt, height=512, width=3072).images[0]
```

The model supports resolutions from 512x512 up to 2048x2048 and beyond.

## Negative prompts

Use negative prompts to guide generation away from unwanted elements:

```python
prompt = "A beautiful portrait photograph"
negative_prompt = "blurry, low quality, distorted, ugly, bad anatomy"

image = pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=40,
    guidance_scale=4.5,
).images[0]
```

## Reproducible generation

Set a seed for reproducible results:

```python
import torch

generator = torch.Generator(device="cuda").manual_seed(42)

# This will always produce the same image
image = pipeline(
    prompt="A red sports car",
    generator=generator,
    num_inference_steps=30,
).images[0]
```

## Batch generation

Generate multiple images at once:

```python
prompts = [
    "A cat sitting on a windowsill",
    "A dog playing in a park",
    "A bird flying in the sky",
]

images = pipeline(
    prompt=prompts,
    num_inference_steps=30,
).images

for i, image in enumerate(images):
    image.save(f"image_{i}.png")
```

## Memory optimization

For systems with limited VRAM, enable CPU offloading:

```python
pipeline.enable_model_cpu_offload()

# Now you can generate images with lower memory requirements
image = pipeline(prompt).images[0]
```

You can also use attention slicing for additional memory savings:

```python
pipeline.enable_attention_slicing()
```

Or use `torch.compile` for faster generation (PyTorch 2.0+):

```python
pipeline.transformer = torch.compile(
    pipeline.transformer,
    mode="reduce-overhead",
    fullgraph=True
)
```

## Advanced: Custom text embeddings

You can pass pre-computed text embeddings instead of prompts:

```python
from transformers import AutoTokenizer, AutoModel

# Get text embeddings
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
text_encoder = AutoModel.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.bfloat16
).to("cuda")

# Encode prompt
inputs = tokenizer(
    "A beautiful landscape",
    return_tensors="pt",
    padding="max_length",
    max_length=128,
    truncation=True
).to("cuda")

prompt_embeds = text_encoder(**inputs).last_hidden_state
prompt_attention_mask = inputs.attention_mask

# Generate with embeddings
image = pipeline(
    prompt_embeds=prompt_embeds,
    prompt_attention_mask=prompt_attention_mask,
    num_inference_steps=30,
).images[0]
```

## Comparison with Lumina-Next

If you're choosing between Lumina-T2I and Lumina-Next:

| Feature      | Lumina-T2I              | Lumina-Next        |
| ------------ | ----------------------- | ------------------ |
| Text Encoder | LLaMA-2-7B              | Gemma              |
| Architecture | DiT-Llama               | NextDiT (improved) |
| Speed        | Baseline                | ~20% faster        |
| Quality      | High                    | Improved           |
| Use Case     | Original implementation | Enhanced version   |

Use Lumina-T2I for:

- Research and comparison with the original paper
- When you specifically need LLaMA-2 as the text encoder
- Maximum compatibility with the original implementation

Use Lumina-Next for:

- Production deployments
- When you need the best quality and speed
- Latest improvements and features

## Troubleshooting

### Out of memory errors

If you encounter OOM errors:

1. Enable CPU offloading: `pipeline.enable_model_cpu_offload()`
2. Reduce resolution: Use 512x512 instead of 1024x1024
3. Use lower precision: `torch.float16` instead of `torch.bfloat16`
4. Reduce batch size to 1
5. Enable attention slicing: `pipeline.enable_attention_slicing()`

### Slow generation

To speed up generation:

1. Use fewer inference steps (20-25 instead of 30-40)
2. Compile the model with `torch.compile()`
3. Ensure you're using a GPU with sufficient VRAM
4. Use `torch.bfloat16` precision

### Quality issues

For better quality:

1. Increase inference steps to 40-60
2. Adjust guidance scale (try 4.0-5.0)
3. Use higher resolution (1024x1024 or above)
4. Craft more detailed prompts
5. Use negative prompts to exclude unwanted elements

## Resources

- [Lumina-T2I Paper](https://arxiv.org/abs/2405.05945)
- [Original Code Repository](https://github.com/Alpha-VLLM/Lumina-T2X)
- [Model Weights on Hugging Face](https://huggingface.co/Alpha-VLLM/Lumina-T2I)
- [LuminaT2IPipeline API Reference](../api/pipelines/lumina#luminat2ipipeline)
- [LuminaDiT2DModel API Reference](../api/models/lumina_dit2d)
- [LuminaFlowMatchScheduler API Reference](../api/schedulers/lumina_flow_match)
