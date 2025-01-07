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
# limitations under the License.
-->

# Mochi 1 Preview

> [!TIP]
> Only a research preview of the model weights is available at the moment.

[Mochi 1](https://huggingface.co/genmo/mochi-1-preview) is a video generation model by Genmo with a strong focus on prompt adherence and motion quality. The model features a 10B parameter Asmmetric Diffusion Transformer (AsymmDiT) architecture, and uses non-square QKV and output projection layers to reduce inference memory requirements. A single T5-XXL model is used to encode prompts.

*Mochi 1 preview is an open state-of-the-art video generation model with high-fidelity motion and strong prompt adherence in preliminary evaluation. This model dramatically closes the gap between closed and open video generation systems. The model is released under a permissive Apache 2.0 license.*

> [!TIP]
> Make sure to check out the Schedulers [guide](../../using-diffusers/schedulers) to learn how to explore the tradeoff between scheduler speed and quality, and see the [reuse components across pipelines](../../using-diffusers/loading#reuse-a-pipeline) section to learn how to efficiently load the same components into multiple pipelines.

## Quantization

Quantization helps reduce the memory requirements of very large models by storing model weights in a lower precision data type. However, quantization may have varying impact on video quality depending on the video model.

Refer to the [Quantization](../../quantization/overview) overview to learn more about supported quantization backends and selecting a quantization backend that supports your use case. The example below demonstrates how to load a quantized [`MochiPipeline`] for inference with bitsandbytes.

```py
import torch
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig, MochiTransformer3DModel, MochiPipeline
from diffusers.utils import export_to_video
from transformers import BitsAndBytesConfig as BitsAndBytesConfig, T5EncoderModel

quant_config = BitsAndBytesConfig(load_in_8bit=True)
text_encoder_8bit = T5EncoderModel.from_pretrained(
    "genmo/mochi-1-preview",
    subfolder="text_encoder",
    quantization_config=quant_config,
    torch_dtype=torch.float16,
)

quant_config = DiffusersBitsAndBytesConfig(load_in_8bit=True)
transformer_8bit = MochiTransformer3DModel.from_pretrained(
    "genmo/mochi-1-preview",
    subfolder="transformer",
    quantization_config=quant_config,
    torch_dtype=torch.float16,
)

pipeline = MochiPipeline.from_pretrained(
    "genmo/mochi-1-preview",
    text_encoder=text_encoder_8bit,
    transformer=transformer_8bit,
    torch_dtype=torch.float16,
    device_map="balanced",
)

video = pipeline(
  "Close-up of a cats eye, with the galaxy reflected in the cats eye. Ultra high resolution 4k.",
  num_inference_steps=28,
  guidance_scale=3.5
).frames[0]
export_to_video(video, "cat.mp4")
```

## Generating videos with Mochi-1 Preview

The following example will download the full precision `mochi-1-preview` weights and produce the highest quality results but will require at least 42GB VRAM to run.

```python
import torch
from diffusers import MochiPipeline
from diffusers.utils import export_to_video

pipe = MochiPipeline.from_pretrained("genmo/mochi-1-preview")

# Enable memory savings
pipe.enable_model_cpu_offload()
pipe.enable_vae_tiling()

prompt = "Close-up of a chameleon's eye, with its scaly skin changing color. Ultra high resolution 4k."

with torch.autocast("cuda", torch.bfloat16, cache_enabled=False):
      frames = pipe(prompt, num_frames=85).frames[0]

export_to_video(frames, "mochi.mp4", fps=30)
```

## Using a lower precision variant to save memory

The following example will use the `bfloat16` variant of the model and requires 22GB VRAM to run. There is a slight drop in the quality of the generated video as a result.

```python
import torch
from diffusers import MochiPipeline
from diffusers.utils import export_to_video

pipe = MochiPipeline.from_pretrained("genmo/mochi-1-preview", variant="bf16", torch_dtype=torch.bfloat16)

# Enable memory savings
pipe.enable_model_cpu_offload()
pipe.enable_vae_tiling()

prompt = "Close-up of a chameleon's eye, with its scaly skin changing color. Ultra high resolution 4k."
frames = pipe(prompt, num_frames=85).frames[0]

export_to_video(frames, "mochi.mp4", fps=30)
```

## Reproducing the results from the Genmo Mochi repo

The [Genmo Mochi implementation](https://github.com/genmoai/mochi/tree/main) uses different precision values for each stage in the inference process. The text encoder and VAE use `torch.float32`, while the DiT uses `torch.bfloat16` with the [attention kernel](https://pytorch.org/docs/stable/generated/torch.nn.attention.sdpa_kernel.html#torch.nn.attention.sdpa_kernel) set to `EFFICIENT_ATTENTION`. Diffusers pipelines currently do not support setting different `dtypes` for different stages of the pipeline. In order to run inference in the same way as the the original implementation, please refer to the following example.

<Tip>
The original Mochi implementation zeros out empty prompts. However, enabling this option and placing the entire pipeline under autocast can lead to numerical overflows with the T5 text encoder.

When enabling `force_zeros_for_empty_prompt`, it is recommended to run the text encoding step outside the autocast context in full precision.
</Tip>

<Tip>
Decoding the latents in full precision is very memory intensive. You will need at least 70GB VRAM to generate the 163 frames in this example. To reduce memory, either reduce the number of frames or run the decoding step in `torch.bfloat16`.
</Tip>

```python
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from diffusers import MochiPipeline
from diffusers.utils import export_to_video
from diffusers.video_processor import VideoProcessor

pipe = MochiPipeline.from_pretrained("genmo/mochi-1-preview", force_zeros_for_empty_prompt=True)
pipe.enable_vae_tiling()
pipe.enable_model_cpu_offload()

prompt =  "An aerial shot of a parade of elephants walking across the African savannah. The camera showcases the herd and the surrounding landscape."

with torch.no_grad():
    prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask = (
        pipe.encode_prompt(prompt=prompt)
    )

with torch.autocast("cuda", torch.bfloat16):
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        frames = pipe(
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            guidance_scale=4.5,
            num_inference_steps=64,
            height=480,
            width=848,
            num_frames=163,
            generator=torch.Generator("cuda").manual_seed(0),
            output_type="latent",
            return_dict=False,
        )[0]

video_processor = VideoProcessor(vae_scale_factor=8)
has_latents_mean = hasattr(pipe.vae.config, "latents_mean") and pipe.vae.config.latents_mean is not None
has_latents_std = hasattr(pipe.vae.config, "latents_std") and pipe.vae.config.latents_std is not None
if has_latents_mean and has_latents_std:
    latents_mean = (
        torch.tensor(pipe.vae.config.latents_mean).view(1, 12, 1, 1, 1).to(frames.device, frames.dtype)
    )
    latents_std = (
        torch.tensor(pipe.vae.config.latents_std).view(1, 12, 1, 1, 1).to(frames.device, frames.dtype)
    )
    frames = frames * latents_std / pipe.vae.config.scaling_factor + latents_mean
else:
    frames = frames / pipe.vae.config.scaling_factor

with torch.no_grad():
    video = pipe.vae.decode(frames.to(pipe.vae.dtype), return_dict=False)[0]

video = video_processor.postprocess_video(video)[0]
export_to_video(video, "mochi.mp4", fps=30)
```

## Running inference with multiple GPUs

It is possible to split the large Mochi transformer across multiple GPUs using the `device_map` and `max_memory` options in `from_pretrained`. In the following example we split the model across two GPUs, each with 24GB of VRAM.

```python
import torch
from diffusers import MochiPipeline, MochiTransformer3DModel
from diffusers.utils import export_to_video

model_id = "genmo/mochi-1-preview"
transformer = MochiTransformer3DModel.from_pretrained(
    model_id,
    subfolder="transformer",
    device_map="auto",
    max_memory={0: "24GB", 1: "24GB"}
)

pipe = MochiPipeline.from_pretrained(model_id,  transformer=transformer)
pipe.enable_model_cpu_offload()
pipe.enable_vae_tiling()

with torch.autocast(device_type="cuda", dtype=torch.bfloat16, cache_enabled=False):
    frames = pipe(
        prompt="Close-up of a chameleon's eye, with its scaly skin changing color. Ultra high resolution 4k.",
        negative_prompt="",
        height=480,
        width=848,
        num_frames=85,
        num_inference_steps=50,
        guidance_scale=4.5,
        num_videos_per_prompt=1,
        generator=torch.Generator(device="cuda").manual_seed(0),
        max_sequence_length=256,
        output_type="pil",
    ).frames[0]

export_to_video(frames, "output.mp4", fps=30)
```

## Using single file loading with the Mochi Transformer

You can use `from_single_file` to load the Mochi transformer in its original format.

<Tip>
Diffusers currently doesn't support using the FP8 scaled versions of the Mochi single file checkpoints.
</Tip>

```python
import torch
from diffusers import MochiPipeline, MochiTransformer3DModel
from diffusers.utils import export_to_video

model_id = "genmo/mochi-1-preview"

ckpt_path = "https://huggingface.co/Comfy-Org/mochi_preview_repackaged/blob/main/split_files/diffusion_models/mochi_preview_bf16.safetensors"

transformer = MochiTransformer3DModel.from_pretrained(ckpt_path, torch_dtype=torch.bfloat16)

pipe = MochiPipeline.from_pretrained(model_id,  transformer=transformer)
pipe.enable_model_cpu_offload()
pipe.enable_vae_tiling()

with torch.autocast(device_type="cuda", dtype=torch.bfloat16, cache_enabled=False):
    frames = pipe(
        prompt="Close-up of a chameleon's eye, with its scaly skin changing color. Ultra high resolution 4k.",
        negative_prompt="",
        height=480,
        width=848,
        num_frames=85,
        num_inference_steps=50,
        guidance_scale=4.5,
        num_videos_per_prompt=1,
        generator=torch.Generator(device="cuda").manual_seed(0),
        max_sequence_length=256,
        output_type="pil",
    ).frames[0]

export_to_video(frames, "output.mp4", fps=30)
```

## MochiPipeline

[[autodoc]] MochiPipeline
  - all
  - __call__

## MochiPipelineOutput

[[autodoc]] pipelines.mochi.pipeline_output.MochiPipelineOutput
