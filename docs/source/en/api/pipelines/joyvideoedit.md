<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# JoyAI-Video-Edit

[JoyAI-Video-Edit](https://github.com/jd-opensource/JoyAI-Video-Edit) is an instruction-guided video-editing model built on the JoyAI streaming architecture. The source video is VAE-encoded into a latent sequence that conditions a dual-stream MM-DiT transformer, which denoises the edited output one causal chunk at a time. Each chunk attends to a sliding window of previously-denoised chunks (and an optional static reference image) through a per-layer KV cache, keeping later chunks temporally consistent with earlier ones without recomputing their key/value projections.

| Model | Description | Download |
|:-----:|:-----------:|:--------:|
| JoyAI-Video-Edit | Instruction-guided causal video editing | [Hugging Face](https://huggingface.co/jdopensource/JoyAI-Video-Edit-Diffusers) |

```python
import torch
from diffusers import JoyVideoEditPipeline
from diffusers.utils import export_to_video, load_video
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

mimo_id = "XiaomiMiMo/MiMo-VL-7B-RL-2508"
processor = AutoProcessor.from_pretrained(mimo_id)
text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    mimo_id, dtype=torch.bfloat16
)

pipeline = JoyVideoEditPipeline.from_pretrained(
    "jdopensource/JoyAI-Video-Edit-Diffusers",
    text_encoder=text_encoder,
    processor=processor,
    dtype=torch.bfloat16,
)
pipeline.enable_model_cpu_offload()

video = load_video("https://raw.githubusercontent.com/jd-opensource/JoyAI-Video-Edit/main/assets/input.mp4")
prompt = (
    "Transform the scene into a British castle royal aristocratic style. Modify the characters' clothing to "
    "aristocratic attire: dress the man in a tailored velvet suit with a ruffled cravat, and the women in elegant "
    "silk gowns with lace details and embroidered bodices. Change their hairstyles to classic aristocratic styles, "
    "such as elaborate updos with subtle jewels for the women and a neatly styled classic cut for the man. Change "
    "the environmental decoration to a British castle interior: replace the plain walls and abstract painting with "
    "stone walls and antique oil paintings in gilded frames, and replace the white window curtains with heavy velvet "
    "drapes. The characters' ages and facial features must remain completely unchanged. The dining table, white "
    "tablecloth, plates of food, wine glasses, water glasses, and the characters' positions and actions must remain "
    "unchanged."
)

output = pipeline(
    video=video,
    prompt=prompt,
    num_inference_steps=2,
    generator=torch.Generator(device="cpu").manual_seed(0),
)
export_to_video(output.frames[0], "joyvideoedit_output.mp4", fps=24)
```

The pipeline denoises with a flow-matching scheduler and does not use classifier-free guidance, so it takes neither a
`negative_prompt` nor a `guidance_scale` argument. An optional static reference image can be supplied through
`ref_image`; its KV is prefilled into the cache and attended to by every chunk to inject appearance conditioning.

The Diffusers checkpoint does not include MiMo-VL. Load [`XiaomiMiMo/MiMo-VL-7B-RL-2508`](https://huggingface.co/XiaomiMiMo/MiMo-VL-7B-RL-2508) from MiMo-VL's own repository
and pass its model and processor to [`JoyVideoEditPipeline.from_pretrained`]. The tokenizer from the processor is used
when a separate `tokenizer` is not provided. You can omit all MiMo-VL components when passing precomputed
`prompt_embeds` and `prompt_embeds_mask`.

Model CPU offloading is recommended because the transformer, MiMo-VL, and VAE are otherwise resident on the GPU at
the same time. The pipeline also supports sequential CPU offloading for lower memory use and pipeline-level group
offloading for a balance between transfer overhead and memory use.

## JoyVideoEditPipeline

[[autodoc]] JoyVideoEditPipeline
  - all
  - __call__

## JoyVideoEditPipelineOutput

[[autodoc]] pipelines.joyvideoedit.pipeline_output.JoyVideoEditPipelineOutput
