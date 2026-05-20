# HiDream-O1

HiDream-O1 is a Qwen3-VL based image generation model that predicts raw RGB image patches directly. Unlike HiDream-I1,
it does not use a VAE component.

The following models are supported by [`HiDreamO1ModularPipeline`]:

| Model | Hugging Face Hub |
|---|---|
| HiDream-O1-Image | [`HiDream-ai/HiDream-O1-Image`](https://huggingface.co/HiDream-ai/HiDream-O1-Image) |
| HiDream-O1-Image-Dev | [`HiDream-ai/HiDream-O1-Image-Dev`](https://huggingface.co/HiDream-ai/HiDream-O1-Image-Dev) |

```python
import torch
from transformers import AutoProcessor

from diffusers import HiDreamO1ModularPipeline, HiDreamO1Transformer2DModel

processor = AutoProcessor.from_pretrained("HiDream-ai/HiDream-O1-Image")
transformer = HiDreamO1Transformer2DModel.from_pretrained(
    "HiDream-ai/HiDream-O1-Image", torch_dtype=torch.bfloat16
)

pipe = HiDreamO1ModularPipeline()
pipe.update_components(processor=processor, transformer=transformer)
pipe.to("cuda")

image = pipe(
    prompt="A cinematic portrait of a glass astronaut standing in a neon-lit botanical garden.",
    generator=torch.Generator("cuda").manual_seed(32),
).images[0]
```

## HiDreamO1ModularPipeline

[[autodoc]] HiDreamO1ModularPipeline

## HiDreamO1AutoBlocks

[[autodoc]] HiDreamO1AutoBlocks
