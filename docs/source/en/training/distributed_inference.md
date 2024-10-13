<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Distributed inference

On distributed setups, you can run inference across multiple GPUs with 🤗 [Accelerate](https://huggingface.co/docs/accelerate/index) or [PyTorch Distributed](https://pytorch.org/tutorials/beginner/dist_overview.html), which is useful for generating with multiple prompts in parallel.

This guide will show you how to use 🤗 Accelerate and PyTorch Distributed for distributed inference.

## 🤗 Accelerate

🤗 [Accelerate](https://huggingface.co/docs/accelerate/index) is a library designed to make it easy to train or run inference across distributed setups. It simplifies the process of setting up the distributed environment, allowing you to focus on your PyTorch code.

To begin, create a Python file and initialize an [`accelerate.PartialState`] to create a distributed environment; your setup is automatically detected so you don't need to explicitly define the `rank` or `world_size`. Move the [`DiffusionPipeline`] to `distributed_state.device` to assign a GPU to each process.

Now use the [`~accelerate.PartialState.split_between_processes`] utility as a context manager to automatically distribute the prompts between the number of processes.

```py
import torch
from accelerate import PartialState
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16, use_safetensors=True
)
distributed_state = PartialState()
pipeline.to(distributed_state.device)

with distributed_state.split_between_processes(["a dog", "a cat"]) as prompt:
    result = pipeline(prompt).images[0]
    result.save(f"result_{distributed_state.process_index}.png")
```

Use the `--num_processes` argument to specify the number of GPUs to use, and call `accelerate launch` to run the script:

```bash
accelerate launch run_distributed.py --num_processes=2
```

<Tip>

Refer to this minimal example [script](https://gist.github.com/sayakpaul/cfaebd221820d7b43fae638b4dfa01ba) for running inference across multiple GPUs. To learn more, take a look at the [Distributed Inference with 🤗 Accelerate](https://huggingface.co/docs/accelerate/en/usage_guides/distributed_inference#distributed-inference-with-accelerate) guide.

</Tip>

## PyTorch Distributed

PyTorch supports [`DistributedDataParallel`](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) which enables data parallelism.

To start, create a Python file and import `torch.distributed` and `torch.multiprocessing` to set up the distributed process group and to spawn the processes for inference on each GPU. You should also initialize a [`DiffusionPipeline`]:

```py
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from diffusers import DiffusionPipeline

sd = DiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16, use_safetensors=True
)
```

You'll want to create a function to run inference; [`init_process_group`](https://pytorch.org/docs/stable/distributed.html?highlight=init_process_group#torch.distributed.init_process_group) handles creating a distributed environment with the type of backend to use, the `rank` of the current process, and the `world_size` or the number of processes participating. If you're running inference in parallel over 2 GPUs, then the `world_size` is 2.

Move the [`DiffusionPipeline`] to `rank` and use `get_rank` to assign a GPU to each process, where each process handles a different prompt:

```py
def run_inference(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    sd.to(rank)

    if torch.distributed.get_rank() == 0:
        prompt = "a dog"
    elif torch.distributed.get_rank() == 1:
        prompt = "a cat"

    image = sd(prompt).images[0]
    image.save(f"./{'_'.join(prompt)}.png")
```

To run the distributed inference, call [`mp.spawn`](https://pytorch.org/docs/stable/multiprocessing.html#torch.multiprocessing.spawn) to run the `run_inference` function on the number of GPUs defined in `world_size`:

```py
def main():
    world_size = 2
    mp.spawn(run_inference, args=(world_size,), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
```

Once you've completed the inference script, use the `--nproc_per_node` argument to specify the number of GPUs to use and call `torchrun` to run the script:

```bash
torchrun run_distributed.py --nproc_per_node=2
```

> [!TIP]
> You can use `device_map` within a [`DiffusionPipeline`] to distribute its model-level components on multiple devices. Refer to the [Device placement](../tutorials/inference_with_big_models#device-placement) guide to learn more.

## Model sharding

Modern diffusion systems such as [Flux](../api/pipelines/flux) are very large and have multiple models. For example, [Flux.1-Dev](https://hf.co/black-forest-labs/FLUX.1-dev) is made up of two text encoders - [T5-XXL](https://hf.co/google/t5-v1_1-xxl) and [CLIP-L](https://hf.co/openai/clip-vit-large-patch14) - a [diffusion transformer](../api/models/flux_transformer), and a [VAE](../api/models/autoencoderkl). With a model this size, it can be challenging to run inference on consumer GPUs.

Model sharding is a technique that distributes models across GPUs when the models don't fit on a single GPU. The example below assumes two 16GB GPUs are available for inference.

Start by computing the text embeddings with the text encoders. Keep the text encoders on two GPUs by setting `device_map="balanced"`. The `balanced` strategy evenly distributes the model on all available GPUs. Use the `max_memory` parameter to allocate the maximum amount of memory for each text encoder on each GPU.

> [!TIP]
> **Only** load the text encoders for this step! The diffusion transformer and VAE are loaded in a later step to preserve memory.

```py
from diffusers import FluxPipeline
import torch

prompt = "a photo of a dog with cat-like look"

pipeline = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    transformer=None,
    vae=None,
    device_map="balanced",
    max_memory={0: "16GB", 1: "16GB"},
    torch_dtype=torch.bfloat16
)
with torch.no_grad():
    print("Encoding prompts.")
    prompt_embeds, pooled_prompt_embeds, text_ids = pipeline.encode_prompt(
        prompt=prompt, prompt_2=None, max_sequence_length=512
    )
```

Once the text embeddings are computed, remove them from the GPU to make space for the diffusion transformer.

```py
import gc 

def flush():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()

del pipeline.text_encoder
del pipeline.text_encoder_2
del pipeline.tokenizer
del pipeline.tokenizer_2
del pipeline

flush()
```

Load the diffusion transformer next which has 12.5B parameters. This time, set `device_map="auto"` to automatically distribute the model across two 16GB GPUs. The `auto` strategy is backed by [Accelerate](https://hf.co/docs/accelerate/index) and available as a part of the [Big Model Inference](https://hf.co/docs/accelerate/concept_guides/big_model_inference) feature. It starts by distributing a model across the fastest device first (GPU) before moving to slower devices like the CPU and hard drive if needed. The trade-off of storing model parameters on slower devices is slower inference latency.

```py
from diffusers import FluxTransformer2DModel
import torch 

transformer = FluxTransformer2DModel.from_pretrained(
    "black-forest-labs/FLUX.1-dev", 
    subfolder="transformer",
    device_map="auto",
    torch_dtype=torch.bfloat16
)
```

> [!TIP]
> At any point, you can try `print(pipeline.hf_device_map)` to see how the various models are distributed across devices. This is useful for tracking the device placement of the models. You can also try `print(transformer.hf_device_map)` to see how the transformer model is sharded across devices.

Add the transformer model to the pipeline for denoising, but set the other model-level components like the text encoders and VAE to `None` because you don't need them yet.

```py
pipeline = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", ,
    text_encoder=None,
    text_encoder_2=None,
    tokenizer=None,
    tokenizer_2=None,
    vae=None,
    transformer=transformer,
    torch_dtype=torch.bfloat16
)

print("Running denoising.")
height, width = 768, 1360
latents = pipeline(
    prompt_embeds=prompt_embeds,
    pooled_prompt_embeds=pooled_prompt_embeds,
    num_inference_steps=50,
    guidance_scale=3.5,
    height=height,
    width=width,
    output_type="latent",
).images
```

Remove the pipeline and transformer from memory as they're no longer needed.

```py
del pipeline.transformer
del pipeline

flush()
```

Finally, decode the latents with the VAE into an image. The VAE is typically small enough to be loaded on a single GPU.

```py
from diffusers import AutoencoderKL
from diffusers.image_processor import VaeImageProcessor
import torch 

vae = AutoencoderKL.from_pretrained(ckpt_id, subfolder="vae", torch_dtype=torch.bfloat16).to("cuda")
vae_scale_factor = 2 ** (len(vae.config.block_out_channels))
image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)

with torch.no_grad():
    print("Running decoding.")
    latents = FluxPipeline._unpack_latents(latents, height, width, vae_scale_factor)
    latents = (latents / vae.config.scaling_factor) + vae.config.shift_factor

    image = vae.decode(latents, return_dict=False)[0]
    image = image_processor.postprocess(image, output_type="pil")
    image[0].save("split_transformer.png")
```

By selectively loading and unloading the models you need at a given stage and sharding the largest models across multiple GPUs, it is possible to run inference with large models on consumer GPUs.
