# LLaDA-Image

[LLaDA-Image](https://huggingface.co/inclusionAI/LLaDA-Image) is a unified image generation and editing model. The
same pipeline supports text-to-image generation, VQ-conditioned generation, and instruction-guided editing with a
reference image. The Base checkpoint is designed for 50 sampling steps, while
[LLaDA-Image-Turbo](https://huggingface.co/inclusionAI/LLaDA-Image-Turbo) is distilled for 4 steps.

The checkpoint includes a custom LLaDA2 text encoder, so pass `trust_remote_code=True` when loading it.

```python
import torch

from diffusers import LLaDAImagePipeline

pipe = LLaDAImagePipeline.from_pretrained(
    "inclusionAI/LLaDA-Image",
    dtype=torch.bfloat16,
    trust_remote_code=True,
)
pipe.enable_model_cpu_offload()

image = pipe(
    prompt="A cinematic photograph of a red fox standing in fresh snow",
    height=1024,
    width=1024,
    num_inference_steps=50,
    guidance_scale=5.0,
    generator=torch.Generator("cuda").manual_seed(42),
).images[0]
```

For image editing, pass a reference image and select the editing mode.

```python
from diffusers.utils import load_image

reference_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png")
image = pipe(
    prompt="Turn it into a watercolor painting",
    image=reference_image,
    generation_mode="editing",
    height=1024,
    width=1024,
    num_inference_steps=50,
    guidance_scale=5.0,
).images[0]
```

## LLaDAImagePipeline

[[autodoc]] LLaDAImagePipeline
  - all
  - call

## LLaDAImagePipelineOutput

[[autodoc]] pipelines.LLaDAImagePipelineOutput
