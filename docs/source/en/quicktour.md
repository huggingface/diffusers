<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Quickstart

Diffusers is a library for developers and researchers that provides an easy inference API for generating images, videos and audio, as well as the building blocks for implementing new workflows.

Diffusers provides many optimizations out-of-the-box that make it possible to load and run large models on setups with limited memory or to accelerate inference.

This Quickstart will give you an overview of Diffusers and get you up and generating quickly.

> [!TIP]
> Before you begin, make sure you have a Hugging Face [account](https://huggingface.co/join) to use gated models like [Flux](https://huggingface.co/black-forest-labs/FLUX.1-dev). Building a custom pipeline instead? See the [Modular Diffusers quickstart](./modular_diffusers/quickstart) and [overview](./modular_diffusers/overview).

Follow the [Installation](./installation) guide to install Diffusers if it's not already installed.

## Agent prompt

Paste this into your coding agent to get Diffusers set up for inference.

```text
Help me get set up with Hugging Face Diffusers for inference.

1. Install Diffusers for my environment with `uv pip install "diffusers[torch]"`.
2. If I need gated Hub models, help me authenticate to the Hugging Face Hub.
3. Install Diffusers coding agent skills with `diffusers-cli skills add diffusers-cli`. Pass `--cursor`, `--claude`, or `--codex` if auto-detect fails. Optionally pass `--all` to install every skill in the registry.
4. Run a first text-to-image with DiffusionPipeline or `diffusers-cli run`, using a small or current Quickstart model and the right `device_map` for my machine.
5. Ask what I want next and point me at the matching docs.
```

## DiffusionPipeline

[`DiffusionPipeline`] packages the pieces of a diffusion model (text encoder, scheduler, UNet or DiT, and VAE) into one class for inference. Load with [`~DiffusionPipeline.from_pretrained`], then call the pipeline.

```text
prompt -> text encoder -> embeddings -+-> UNet/DiT <- scheduler  ====xN====
noise --------------------------------+        |
                                            latents -> VAE -> image
```

Arguments on [`~DiffusionPipeline.__call__`] such as `num_inference_steps` change quality and speed. For loading details and mix-and-match components, see [Load pipelines](./using-diffusers/loading). To swap the scheduler, see [Schedulers](./using-diffusers/schedulers).

The examples below use the default argument values.

<hfoptions id="diffusionpipeline">
<hfoption id="text-to-image">

Use `.images[0]` to access the generated image output.

```py
import torch
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
  "Qwen/Qwen-Image", dtype=torch.bfloat16, device_map="cuda"  # or "mps", "xpu", "cpu"
)

prompt = """
cinematic film still of a cat sipping a margarita in a pool in Palm Springs, California
highly detailed, high budget hollywood movie, cinemascope, moody, epic, gorgeous, film grain
"""
pipeline(prompt).images[0]
```

</hfoption>
<hfoption id="text-to-video">

Use `.frames[0]` to access the generated video output and [`~utils.export_to_video`] to save the video.

```py
import torch
from diffusers import AutoencoderKLWan, DiffusionPipeline
from diffusers.utils import export_to_video

vae = AutoencoderKLWan.from_pretrained(
  "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
  subfolder="vae",
  dtype=torch.float32
)
pipeline = DiffusionPipeline.from_pretrained(
  "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
  vae=vae,
  dtype=torch.bfloat16,
  device_map="cuda"  # or "mps", "xpu", "cpu"
)

prompt = """
Cinematic video of a sleek cat lounging on a colorful inflatable in a crystal-clear turquoise pool in Palm Springs, 
sipping a salt-rimmed margarita through a straw. Golden-hour sunlight glows over mid-century modern homes and swaying palms. 
Shot in rich Sony a7S III: with moody, glamorous color grading, subtle lens flares, and soft vintage film grain. 
Ripples shimmer as a warm desert breeze stirs the water, blending luxury and playful charm in an epic, gorgeously composed frame.
"""
video = pipeline(prompt=prompt, num_frames=81, num_inference_steps=40).frames[0]
export_to_video(video, "output.mp4", fps=16)
```

</hfoption>
</hfoptions>

## LoRA

[LoRA](./tutorials/using_peft_for_inference) adapters add a small style or subject checkpoint on top of a base pipeline. Load one with [`~loaders.QwenImageLoraLoaderMixin.load_lora_weights`]. Some LoRAs need a trigger phrase. Check the LoRA's model card.

```py
import torch
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
  "Qwen/Qwen-Image", dtype=torch.bfloat16, device_map="cuda"  # or "mps", "xpu", "cpu"
)
pipeline.load_lora_weights(
  "threecrowco/VolkClipartQwen",
  weight_name="pytorch_lora_weights.safetensors",
)

prompt = """
Volk clipart style drawing of a cat sipping a margarita in a pool in Palm Springs, California, flat colors, bold outlines, simple shapes
"""
pipeline(prompt).images[0]
```

## Quantization and optimizations

Large models often need less memory or more speed. Use [quantization](./quantization/overview) to shrink weights in memory, and [`~ModelMixin.compile_repeated_blocks`] to speed up later generates. For [model offloading](./optimization/memory#model-offloading) and other options, see [Optimize and scale](./stable_diffusion).

To use less memory, load in 4-bit with bitsandbytes.

```py
import torch
from diffusers import DiffusionPipeline
from diffusers.quantizers import PipelineQuantizationConfig

quant_config = PipelineQuantizationConfig(
  quant_backend="bitsandbytes_4bit",
  quant_kwargs={"load_in_4bit": True, "bnb_4bit_quant_type": "nf4", "bnb_4bit_compute_dtype": torch.bfloat16},
  components_to_quantize=["transformer", "text_encoder"],
)
pipeline = DiffusionPipeline.from_pretrained(
  "Qwen/Qwen-Image",
  dtype=torch.bfloat16,
  quantization_config=quant_config,
  device_map="cuda"  # or "mps", "xpu", "cpu"
)

prompt = """
cinematic film still of a cat sipping a margarita in a pool in Palm Springs, California
highly detailed, high budget hollywood movie, cinemascope, moody, epic, gorgeous, film grain
"""
pipeline(prompt).images[0]
```

To speed up later runs, compile repeated blocks on the transformer. The first generate after compile is a cold start and is slow. Later generates are faster.

```py
import torch
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
  "Qwen/Qwen-Image", dtype=torch.bfloat16, device_map="cuda"  # or "mps", "xpu", "cpu"
)
pipeline.transformer.compile_repeated_blocks(fullgraph=True)

prompt = """
cinematic film still of a cat sipping a margarita in a pool in Palm Springs, California
highly detailed, high budget hollywood movie, cinemascope, moody, epic, gorgeous, film grain
"""
pipeline(prompt).images[0]
```

## Next steps

- [Inference](./using-diffusers/loading) — pipelines, prompting, and adapters
- [Optimize and scale](./stable_diffusion) — memory, speed, quantization, and serving
- [Modular Diffusers](./modular_diffusers/overview) — composable blocks and custom pipelines
- [Train and fine-tune](./training/overview) — training scripts and adapters
- [CLI](./using-diffusers/cli) — generate from the command line
