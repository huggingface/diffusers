<!-- Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License. -->

# LTX Video

[LTX Video](https://huggingface.co/Lightricks/LTX-Video) is the first DiT-based video generation model capable of generating high-quality videos in real-time. It produces 24 FPS videos at a 768x512 resolution faster than they can be watched. Trained on a large-scale dataset of diverse videos, the model generates high-resolution videos with realistic and varied content. We provide a model for both text-to-video as well as image + text-to-video usecases.

<Tip>

Make sure to check out the Schedulers [guide](../../using-diffusers/schedulers) to learn how to explore the tradeoff between scheduler speed and quality, and see the [reuse components across pipelines](../../using-diffusers/loading#reuse-a-pipeline) section to learn how to efficiently load the same components into multiple pipelines.

</Tip>

Available models:

|  Model name   | Recommended dtype |
|:-------------:|:-----------------:|
| [`LTX Video 0.9.0`](https://huggingface.co/Lightricks/LTX-Video/blob/main/ltx-video-2b-v0.9.safetensors) | `torch.bfloat16` |
| [`LTX Video 0.9.1`](https://huggingface.co/Lightricks/LTX-Video/blob/main/ltx-video-2b-v0.9.1.safetensors) | `torch.bfloat16` |

Note: The recommended dtype is for the transformer component. The VAE and text encoders can be either `torch.float32`, `torch.bfloat16` or `torch.float16` but the recommended dtype is `torch.bfloat16` as used in the original repository.

## Loading Single Files

Loading the original LTX Video checkpoints is also possible with [`~ModelMixin.from_single_file`]. We recommend using `from_single_file` for the Lightricks series of models, as they plan to release multiple models in the future in the single file format.

```python
import torch
from diffusers import AutoencoderKLLTXVideo, LTXImageToVideoPipeline, LTXVideoTransformer3DModel

# `single_file_url` could also be https://huggingface.co/Lightricks/LTX-Video/ltx-video-2b-v0.9.1.safetensors
single_file_url = "https://huggingface.co/Lightricks/LTX-Video/ltx-video-2b-v0.9.safetensors"
transformer = LTXVideoTransformer3DModel.from_single_file(
  single_file_url, torch_dtype=torch.bfloat16
)
vae = AutoencoderKLLTXVideo.from_single_file(single_file_url, torch_dtype=torch.bfloat16)
pipe = LTXImageToVideoPipeline.from_pretrained(
  "Lightricks/LTX-Video", transformer=transformer, vae=vae, torch_dtype=torch.bfloat16
)

# ... inference code ...
```

Alternatively, the pipeline can be used to load the weights with [`~FromSingleFileMixin.from_single_file`].

```python
import torch
from diffusers import LTXImageToVideoPipeline
from transformers import T5EncoderModel, T5Tokenizer

single_file_url = "https://huggingface.co/Lightricks/LTX-Video/ltx-video-2b-v0.9.safetensors"
text_encoder = T5EncoderModel.from_pretrained(
  "Lightricks/LTX-Video", subfolder="text_encoder", torch_dtype=torch.bfloat16
)
tokenizer = T5Tokenizer.from_pretrained(
  "Lightricks/LTX-Video", subfolder="tokenizer", torch_dtype=torch.bfloat16
)
pipe = LTXImageToVideoPipeline.from_single_file(
  single_file_url, text_encoder=text_encoder, tokenizer=tokenizer, torch_dtype=torch.bfloat16
)
```

Loading [LTX GGUF checkpoints](https://huggingface.co/city96/LTX-Video-gguf) are also supported:

```py
import torch
from diffusers.utils import export_to_video
from diffusers import LTXPipeline, LTXVideoTransformer3DModel, GGUFQuantizationConfig

ckpt_path = (
    "https://huggingface.co/city96/LTX-Video-gguf/blob/main/ltx-video-2b-v0.9-Q3_K_S.gguf"
)
transformer = LTXVideoTransformer3DModel.from_single_file(
    ckpt_path,
    quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
    torch_dtype=torch.bfloat16,
)
pipe = LTXPipeline.from_pretrained(
    "Lightricks/LTX-Video",
    transformer=transformer,
    torch_dtype=torch.bfloat16,
)
pipe.enable_model_cpu_offload()

prompt = "A woman with long brown hair and light skin smiles at another woman with long blonde hair. The woman with brown hair wears a black jacket and has a small, barely noticeable mole on her right cheek. The camera angle is a close-up, focused on the woman with brown hair's face. The lighting is warm and natural, likely from the setting sun, casting a soft glow on the scene. The scene appears to be real-life footage"
negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"

video = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=704,
    height=480,
    num_frames=161,
    num_inference_steps=50,
).frames[0]
export_to_video(video, "output_gguf_ltx.mp4", fps=24)
```

Make sure to read the [documentation on GGUF](../../quantization/gguf) to learn more about our GGUF support.

<!-- TODO(aryan): Update this when official weights are supported -->

Loading and running inference with [LTX Video 0.9.1](https://huggingface.co/Lightricks/LTX-Video/blob/main/ltx-video-2b-v0.9.1.safetensors) weights.

```python
import torch
from diffusers import LTXPipeline
from diffusers.utils import export_to_video

pipe = LTXPipeline.from_pretrained("a-r-r-o-w/LTX-Video-0.9.1-diffusers", torch_dtype=torch.bfloat16)
pipe.to("cuda")

prompt = "A woman with long brown hair and light skin smiles at another woman with long blonde hair. The woman with brown hair wears a black jacket and has a small, barely noticeable mole on her right cheek. The camera angle is a close-up, focused on the woman with brown hair's face. The lighting is warm and natural, likely from the setting sun, casting a soft glow on the scene. The scene appears to be real-life footage"
negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"

video = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=768,
    height=512,
    num_frames=161,
    decode_timestep=0.03,
    decode_noise_scale=0.025,
    num_inference_steps=50,
).frames[0]
export_to_video(video, "output.mp4", fps=24)
```

Refer to [this section](https://huggingface.co/docs/diffusers/main/en/api/pipelines/cogvideox#memory-optimization) to learn more about optimizing memory consumption.

## Quantization

Quantization helps reduce the memory requirements of very large models by storing model weights in a lower precision data type. However, quantization may have varying impact on video quality depending on the video model.

Refer to the [Quantization](../../quantization/overview) overview to learn more about supported quantization backends and selecting a quantization backend that supports your use case. The example below demonstrates how to load a quantized [`LTXPipeline`] for inference with bitsandbytes.

```py
import torch
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig, LTXVideoTransformer3DModel, LTXPipeline
from diffusers.utils import export_to_video
from transformers import BitsAndBytesConfig as BitsAndBytesConfig, T5EncoderModel

quant_config = BitsAndBytesConfig(load_in_8bit=True)
text_encoder_8bit = T5EncoderModel.from_pretrained(
    "Lightricks/LTX-Video",
    subfolder="text_encoder",
    quantization_config=quant_config,
    torch_dtype=torch.float16,
)

quant_config = DiffusersBitsAndBytesConfig(load_in_8bit=True)
transformer_8bit = LTXVideoTransformer3DModel.from_pretrained(
    "Lightricks/LTX-Video",
    subfolder="transformer",
    quantization_config=quant_config,
    torch_dtype=torch.float16,
)

pipeline = LTXPipeline.from_pretrained(
    "Lightricks/LTX-Video",
    text_encoder=text_encoder_8bit,
    transformer=transformer_8bit,
    torch_dtype=torch.float16,
    device_map="balanced",
)

prompt = "A detailed wooden toy ship with intricately carved masts and sails is seen gliding smoothly over a plush, blue carpet that mimics the waves of the sea. The ship's hull is painted a rich brown, with tiny windows. The carpet, soft and textured, provides a perfect backdrop, resembling an oceanic expanse. Surrounding the ship are various other toys and children's items, hinting at a playful environment. The scene captures the innocence and imagination of childhood, with the toy ship's journey symbolizing endless adventures in a whimsical, indoor setting."
video = pipeline(prompt=prompt, num_frames=161, num_inference_steps=50).frames[0]
export_to_video(video, "ship.mp4", fps=24)
```

## LTXPipeline

[[autodoc]] LTXPipeline
  - all
  - __call__

## LTXImageToVideoPipeline

[[autodoc]] LTXImageToVideoPipeline
  - all
  - __call__

## LTXPipelineOutput

[[autodoc]] pipelines.ltx.pipeline_output.LTXPipelineOutput
