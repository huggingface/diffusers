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

Diffusers provides many optimizations out-of-the-box that makes it possible to load and run large models on setups with limited memory or to accelerate inference.

This Quickstart will give you an overview of Diffusers and get you up and generating quickly.

> [!TIP]
> Before you begin, make sure you have a Hugging Face [account](https://huggingface.co/join) in order to use gated models like [Flux](https://huggingface.co/black-forest-labs/FLUX.1-dev).

Follow the [Installation](./installation) guide to install Diffusers if it's not already installed.

## DiffusionPipeline

A diffusion model combines multiple components to generate outputs in any modality based on an input, such as a text description, image or both.

For a standard text-to-image model:

1. A text encoder turns a prompt into embeddings that guide the denoising process. Some models have more than one text encoder.
2. A scheduler contains the algorithmic specifics for gradually denoising initial random noise into clean outputs. Different schedulers affect generation speed and quality.
3. A UNet or diffusion transformer (DiT) is the workhorse of a diffusion model.

  At each step, it performs the denoising predictions, such as how much noise to remove or the general direction in which to steer the noise to generate better quality outputs.

  The UNet or DiT repeats this loop for a set amount of steps to generate the final output.
  
4. A variational autoencoder (VAE) encodes and decodes pixels to a spatially compressed latent-space. *Latents* are compressed representations of an image and are more efficient to work with. The UNet or DiT operates on latents, and the clean latents at the end are decoded back into images.

The [`DiffusionPipeline`] packages all these components into a single class for inference. There are several arguments in [`~DiffusionPipeline.__call__`] you can change, such as `num_inference_steps`, that affect the diffusion process. Try different values and arguments to see how they change generation quality or speed.

Load a model with [`~DiffusionPipeline.from_pretrained`] and describe what you'd like to generate. The example below uses the default argument values.

<hfoptions id="diffusionpipeline">
<hfoption id="text-to-image">

Use `.images[0]` to access the generated image output.

```py
import torch
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
  "Qwen/Qwen-Image", torch_dtype=torch.bfloat16, device_map="cuda"
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
from diffusers.quantizers import PipelineQuantizationConfig
from diffusers.utils import export_to_video

vae = AutoencoderKLWan.from_pretrained(
  "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
  subfolder="vae",
  torch_dtype=torch.float32
)
pipeline = DiffusionPipeline.from_pretrained(
  "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
  vae=vae
  torch_dtype=torch.bfloat16,
  device_map="cuda"
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

Adapters insert a small number of trainable parameters to the original base model. Only the inserted parameters are fine-tuned while the rest of the model weights remain frozen. This makes it fast and cheap to fine-tune a model on a new style. Among adapters, [LoRA's](./tutorials/using_peft_for_inference) are the most popular.

Add a LoRA to a pipeline with the [`~loaders.QwenImageLoraLoaderMixin.load_lora_weights`] method. Some LoRA's require a special word to trigger it, such as `Realism`, in the example below. Check a LoRA's model card to see if it requires a trigger word.

```py
import torch
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
  "Qwen/Qwen-Image", torch_dtype=torch.bfloat16, device_map="cuda"
)
pipeline.load_lora_weights(
  "flymy-ai/qwen-image-realism-lora",
)

prompt = """
super Realism cinematic film still of a cat sipping a margarita in a pool in Palm Springs in the style of umempart, California
highly detailed, high budget hollywood movie, cinemascope, moody, epic, gorgeous, film grain
"""
pipeline(prompt).images[0]
```

Check out the [LoRA](./tutorials/using_peft_for_inference) docs or Adapters section to learn more.

## Quantization

[Quantization](./quantization/overview) stores data in fewer bits to reduce memory usage. It may also speed up inference because it takes less time to perform calculations with fewer bits.

Diffusers provides several quantization backends and picking one depends on your use case. For example, [bitsandbytes](./quantization/bitsandbytes) and [torchao](./quantization/torchao) are both simple and easy to use for inference, but torchao supports more [quantization types](./quantization/torchao#supported-quantization-types) like fp8.

Configure [`PipelineQuantizationConfig`] with the backend to use, the specific arguments (refer to the [API](./api/quantization) reference for available arguments) for that backend, and which components to quantize. The example below quantizes the model to 4-bits and only uses 14.93GB of memory.

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
  torch_dtype=torch.bfloat16,
  quantization_config=quant_config,
  device_map="cuda"
)

prompt = """
cinematic film still of a cat sipping a margarita in a pool in Palm Springs, California
highly detailed, high budget hollywood movie, cinemascope, moody, epic, gorgeous, film grain
"""
pipeline(prompt).images[0]
print(f"Max memory reserved: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
```

Take a look at the [Quantization](./quantization/overview) section for more details.

## Optimizations

> [!TIP]
> Optimization is dependent on hardware specs such as memory. Use this [Space](https://huggingface.co/spaces/diffusers/optimized-diffusers-code) to generate code examples that include all of Diffusers' available memory and speed optimization techniques for any model you're using.

Modern diffusion models are very large and have billions of parameters. The iterative denoising process is also computationally intensive and slow. Diffusers provides techniques for reducing memory usage and boosting inference speed. These techniques can be combined with quantization to optimize for both memory usage and inference speed.

### Memory usage

The text encoders and UNet or DiT can use up as much as ~30GB of memory, exceeding the amount available on many free-tier or consumer GPUs.

Offloading stores weights that aren't currently used on the CPU and only moves them to the GPU when they're needed. There are a few offloading types and the example below uses [model offloading](./optimization/memory#model-offloading). This moves an entire model, like a text encoder or transformer, to the CPU when it isn't actively being used.

Call [`~DiffusionPipeline.enable_model_cpu_offload`] to activate it. By combining quantization and offloading, the following example only requires ~12.54GB of memory.

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
  torch_dtype=torch.bfloat16,
  quantization_config=quant_config,
  device_map="cuda"
)
pipeline.enable_model_cpu_offload()

prompt = """
cinematic film still of a cat sipping a margarita in a pool in Palm Springs, California
highly detailed, high budget hollywood movie, cinemascope, moody, epic, gorgeous, film grain
"""
pipeline(prompt).images[0]
print(f"Max memory reserved: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
```

Refer to the [Reduce memory usage](./optimization/memory) docs to learn more about other memory reducing techniques.

### Inference speed

The denoising loop performs a lot of computations and can be slow. Methods like [torch.compile](./optimization/fp16#torchcompile) increases inference speed by compiling the computations into an optimized kernel. Compilation is slow for the first generation but successive generations should be much faster.

The example below uses [regional compilation](./optimization/fp16#regional-compilation) to only compile small regions of a model. It reduces cold-start latency while also providing a runtime speed up.

Call [`~ModelMixin.compile_repeated_blocks`] on the model to activate it.

```py
import torch
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
  "Qwen/Qwen-Image", torch_dtype=torch.bfloat16, device_map="cuda"
)

pipeline.transformer.compile_repeated_blocks(
    fullgraph=True,
)
prompt = """
cinematic film still of a cat sipping a margarita in a pool in Palm Springs, California
highly detailed, high budget hollywood movie, cinemascope, moody, epic, gorgeous, film grain
"""
pipeline(prompt).images[0]
```

Check out the [Accelerate inference](./optimization/fp16) or [Caching](./optimization/cache) docs for more methods that speed up inference.