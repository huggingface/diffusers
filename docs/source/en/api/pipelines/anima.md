# Anima

Anima is a text-to-image model that reuses the [`CosmosTransformer3DModel`] with a Qwen3 text encoder, a T5-token text conditioner, and the [`AutoencoderKLQwenImage`] VAE.

```python
import torch
from diffusers import ModularPipeline

pipe = ModularPipeline.from_pretrained("mrfatso/anima-preview3-diffusers")
pipe.load_components(torch_dtype=torch.bfloat16)
pipe.to("cuda")

image = pipe(prompt="masterpiece, best quality, 1girl, solo, city lights").images[0]
```

## AnimaModularPipeline

[[autodoc]] AnimaModularPipeline

## AnimaAutoBlocks

[[autodoc]] AnimaAutoBlocks

## AnimaTextConditioner

[[autodoc]] AnimaTextConditioner
