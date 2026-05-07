<!-- Copyright 2026 The HuggingFace Team. All rights reserved. -->

# Motif-Video

[Technical Report](https://arxiv.org/abs/2604.16503)

Motif-Video is a 2B parameter diffusion transformer designed for text-to-video and image-to-video generation. It features a three-stage architecture with 12 dual-stream + 16 single-stream + 8 DDT decoder layers, Shared Cross-Attention for stable text-video alignment under long video sequences, T5Gemma2 text encoder, and rectified flow matching for velocity prediction.

<p align="center">
  <img src="https://huggingface.co/Motif-Technologies/Motif-Video-2B/resolve/main/assets/architecture.png" width="90%" alt="Motif-Video architecture"/>
</p>

## Text-to-Video Generation

Use `MotifVideoPipeline` for text-to-video generation:

```python
import torch
from diffusers import MotifVideoPipeline
from diffusers.utils import export_to_video


pipe = MotifVideoPipeline.from_pretrained(
    "Motif-Technologies/Motif-Video-2B",
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")

prompt = "A woman with long brown hair and light skin smiles at another woman with long blonde hair."
negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"

video = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=1280,
    height=736,
    num_frames=121,
    num_inference_steps=50,
).frames[0]
export_to_video(video, "output.mp4", fps=24)
```

## Image-to-Video Generation

Use `MotifVideoImage2VideoPipeline` for image-to-video generation:

```python
import torch
from diffusers import MotifVideoImage2VideoPipeline
from diffusers.utils import export_to_video, load_image


pipe = MotifVideoImage2VideoPipeline.from_pretrained(
    "Motif-Technologies/Motif-Video-2B",
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")

image = load_image("input_image.png")
prompt = "A cinematic scene with vivid colors."
negative_prompt = "worst quality, blurry, jittery, distorted"

video = pipe(
    image=image,
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=1280,
    height=736,
    num_frames=121,
    num_inference_steps=50,
).frames[0]
export_to_video(video, "i2v_output.mp4", fps=24)
```

### Memory-efficient Inference

For GPUs with less than 30GB VRAM (e.g., RTX 4090), use model CPU offloading:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

```python
import torch
from diffusers import MotifVideoPipeline
from diffusers.utils import export_to_video


pipe = MotifVideoPipeline.from_pretrained(
    "Motif-Technologies/Motif-Video-2B",
    torch_dtype=torch.bfloat16,
)
pipe.enable_model_cpu_offload()

prompt = "A woman with long brown hair and light skin smiles at another woman with long blonde hair."
negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"

video = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=1280,
    height=736,
    num_frames=121,
    num_inference_steps=50,
).frames[0]
export_to_video(video, "output.mp4", fps=24)
```

## MotifVideoPipeline

[[autodoc]] MotifVideoPipeline
  - all
  - __call__

## MotifVideoImage2VideoPipeline

[[autodoc]] MotifVideoImage2VideoPipeline
  - all
  - __call__

## MotifVideoPipelineOutput

[[autodoc]] pipelines.motif_video.pipeline_output.MotifVideoPipelineOutput