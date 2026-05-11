# Anima

Anima is a text-to-image model that reuses the [`CosmosTransformer3DModel`] with a Qwen3 text encoder, a T5-token text conditioner, and the [`AutoencoderKLQwenImage`] VAE.

```python
import torch
from diffusers import AnimaPipeline

pipe = AnimaPipeline.from_pretrained("path/to/anima-diffusers", torch_dtype=torch.bfloat16)
pipe.to("cuda")

image = pipe("A cinematic portrait of a woman in a rain-soaked city street").images[0]
```

## AnimaPipeline

[[autodoc]] AnimaPipeline

## AnimaTextConditioner

[[autodoc]] AnimaTextConditioner
