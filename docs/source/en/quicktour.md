<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Quickstart

Diffusers is a library for developers and researchers that provides an easy inference API for generating images and videos as well as the building blocks for implementing new workflows.

Diffusers provides many optimizations out-of-the-box that makes it possible to load and run large models on setups with limited memory or to accelerate inference.

This Quickstart will give you an overview of Diffusers and get you up and generating quickly.

> [!TIP]
> Before you begin, make sure you have a Hugging Face [account](https://huggingface.co/join) in order to use models like [Flux](https://huggingface.co/black-forest-labs/FLUX.1-dev).

Follow the [Installation](./installation) guide to install Diffusers if you haven't already installed it.

## DiffusionPipeline

A diffusion model is actually comprised of several components that work together to generate an image or video that matches your prompt.

1. A text encoder turns your prompt into embeddings which steers the denoising process.
2. A scheduler contains the *denoising* specifics such as how much noise to remove at each step and the step size. Schedulers affect denoising speed and generation quality.
3. A UNet or diffusion transformer is the workhorse of a diffusion model.

  At each step, it takes the noisy input and predicts the noise based on the step and embeddings. The scheduler uses this prediction to subtract the appropriate amount of noise to produce a slightly cleaner image.
  
  The UNet or diffusion transformer repeats this loop for a set amount of steps to produce the final latent output.

4. A decoder (variational autoencoder) converts the *latents* into the actual image or video. Latents are a compressed representation of the input, making it more efficient to work with than the actual pixels.

The [`DiffusionPipeline`] packages all these components into a single class for inference. There are several arguments in [`DiffusionPipeline`] you can change, such as `num_inference_steps`, that affect the diffusion process. Try different values and arguments to see how they change generation quality or speed.

Load a model with [`~DiffusionPipeline.from_pretrained`] and describe what you'd like to generate. The example below uses the default argument values.

<hfoptions id="diffusionpipeline">
<hfoption id="text-to-image">

Use `.images[0]` to access the generated image output.

```py
import torch
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
  "black-forest-labs/FLUX.1-dev",
  torch_dtype=torch.bfloat16
).to("cuda")

prompt = """
cinematic film still of a cat sipping a margarita in a pool in Palm Springs, California
highly detailed, high budget hollywood movie, cinemascope, moody, epic, gorgeous, film grain
"""
pipeline(prompt).images[0]
```

<div class="flex justify-center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/quicktour-diffusion-pipeline.png">
</div>

</hfoption>
<hfoption id="text-to-video">

Use `.frames[0]` to access the generated video output and [`~utils.export_to_video`] to save the video.

```py
import torch
from diffusers import AutoModel, DiffusionPipeline
from diffusers.quantizers import PipelineQuantizationConfig
from diffusers.utils import export_to_video

pipeline_quant_config = PipelineQuantizationConfig(
    quant_backend="bitsandbytes_4bit",
    quant_kwargs={
      "load_in_4bit": True,
      "bnb_4bit_quant_type": "nf4",
      "bnb_4bit_compute_dtype": torch.bfloat16
      },
    components_to_quantize=["transformer"]
)
pipeline = DiffusionPipeline.from_pretrained(
    "hunyuanvideo-community/HunyuanVideo",
    quantization_config=pipeline_quant_config,
    torch_dtype=torch.bfloat16,
).to("cuda")
pipeline.enable_model_cpu_offload()
pipeline.vae.enable_tiling()

prompt = """
Cinematic video of a sleek cat lounging on a colorful inflatable in a crystal-clear turquoise pool in Palm Springs, 
sipping a salt-rimmed margarita through a straw. Golden-hour sunlight glows over mid-century modern homes and swaying palms. 
Shot in rich Sony a7S III: with moody, glamorous color grading, subtle lens flares, and soft vintage film grain. 
Ripples shimmer as a warm desert breeze stirs the water, blending luxury and playful charm in an epic, gorgeously composed frame.
"""
video = pipeline(prompt=prompt, num_frames=61, num_inference_steps=30).frames[0]
export_to_video(video, "output.mp4", fps=15)
```

<div class="flex justify-center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/quicktour-diffusion-pipeline-video.gif">
</div>

</hfoption>
</hfoptions>

## LoRA

Adapters insert a small number of trainable parameters to the original base model. Only the inserted parameters are fine-tuned while the rest of the model weights remain frozen. This makes it fast and cheap to fine-tune a model on a new style. Among adapters, [LoRA's](./tutorials/using_peft_for_inference) are the most popular.

Add a LoRA to your pipeline with the [`~loaders.FluxLoraLoaderMixin.load_lora_weights`] method. Some LoRA's require a special word to trigger it, such as `GHIBSKY style`, in the example below. Check a LoRA's model card to see if it requires a trigger word.

```py
import torch
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
  "black-forest-labs/FLUX.1-dev",
  torch_dtype=torch.bfloat16
)
pipeline.load_lora_weights("aleksa-codes/flux-ghibsky-illustration")
pipeline.to("cuda")

prompt = """
GHIBSKY style cinematic film still of a cat sipping a margarita in a pool in Palm Springs in the style of umempart, California
highly detailed, high budget hollywood movie, cinemascope, moody, epic, gorgeous, film grain
"""
pipeline(prompt).images[0]
```

<div class="flex justify-center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/quicktour-diffusion-pipeline-lora.png">
</div>

Check out the [LoRA](./tutorials/using_peft_for_inference) docs or Adapters section to learn more.

## Quantization

[Quantization](./quantization/overview) stores data in fewer bits to reduce memory usage. It may also speed up inference because it takes less time to perform calculations with fewer bits.

Diffusers provides several quantization backends, and picking one depends on your use case. For example, [bitsandbytes](./quantization/bitsandbytes) and [torchao](./quantization/torchao) are both simple and easy to use for inference, but torchao supports more [quantization types](./quantization/torchao#supported-quantization-types) like fp8.

Configure [`PipelineQuantizationConfig`] with the backend to use, the specific arguments (refer to the [API](./api/quantization) reference for available arguments) for that backend, and which components to quantize. The example below quantizes the model to 4-bits and only uses 14.93GB of memory.

```py
import torch
from diffusers import DiffusionPipeline
from diffusers.quantizers import PipelineQuantizationConfig

quant_config = PipelineQuantizationConfig(
    quant_backend="bitsandbytes_4bit",
    quant_kwargs={"load_in_4bit": True, "bnb_4bit_quant_type": "nf4", "bnb_4bit_compute_dtype": torch.bfloat16},
    components_to_quantize=["transformer", "text_encoder_2"],
)
pipeline = DiffusionPipeline.from_pretrained(
  "black-forest-labs/FLUX.1-dev",
  torch_dtype=torch.bfloat16,
  quantization_config=quant_config,
).to("cuda")

prompt = """
cinematic film still of a cat sipping a margarita in a pool in Palm Springs, California
highly detailed, high budget hollywood movie, cinemascope, moody, epic, gorgeous, film grain
"""
pipeline(prompt).images[0]
print(f"Max memory reserved: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
```

Take a look at the [Quantization](./quantization/overview) section for more details.

## Optimizations

Modern diffusion models are very large and have billions of parameters. The iterative denoising process is also computationally intensive and slow. Diffusers provides techniques for reducing memory usage and boosting inference speed. These techniques can be used together and with quantization to optimize for both memory and inference speed.

### Memory

The text encoders and UNet or diffusion transformer can use up as much as ~30GB of memory, exceeding the amount available on many free-tier or consumer GPUs.

Offloading stores weights that aren't currently used on the CPU and only moves them to the GPU when they're needed. There are a few offloading types and the example below uses [model offloading](./optimization/memory#model-offloading) to move an entire model, like a text encoder or transformer, to the CPU when it isn't being used.

Call [`~DiffusionPipeline.enable_model_cpu_offload`] to activate it on your pipeline. By combining quantization and offloading, the following example only requires ~12.54GB of memory.

```py
import torch
from diffusers import DiffusionPipeline
from diffusers.quantizers import PipelineQuantizationConfig

quant_config = PipelineQuantizationConfig(
    quant_backend="bitsandbytes_4bit",
    quant_kwargs={"load_in_4bit": True, "bnb_4bit_quant_type": "nf4", "bnb_4bit_compute_dtype": torch.bfloat16},
    components_to_quantize=["transformer", "text_encoder_2"],
)
pipeline = DiffusionPipeline.from_pretrained(
  "black-forest-labs/FLUX.1-dev",
  torch_dtype=torch.bfloat16,
  quantization_config=quant_config,
).to("cuda")
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

The denoising loop performs a lot of computations and can be slow. Methods like [torch.compile](./optimization/fp16#torchcompile) increases inference speed by compiling the computations into an optimized kernel. Compilation is slow at first but it should be much faster the next time as long as the input remains the same.

The example below uses [regional compilation](./optimization/fp16#regional-compilation) to only compile small regions of a model. It reduces cold-start latency while also providing a runtime speed up.

Call [`~ModelMixin.compile_repeated_blocks`] on the model to activate it.

```py
import torch
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
  "black-forest-labs/FLUX.1-dev",
  torch_dtype=torch.bfloat16
).to("cuda")

pipeline.transformer.compile_repeated_blocks(
    fullgraph=True, dynamic=True
)
prompt = """
cinematic film still of a cat sipping a margarita in a pool in Palm Springs, California
highly detailed, high budget hollywood movie, cinemascope, moody, epic, gorgeous, film grain
"""
pipeline(prompt).images[0]
```

Check out the [Accelerate inference](./optimization/fp16) or [Caching](./optimization/cache) docs for more methods that speed up inference.