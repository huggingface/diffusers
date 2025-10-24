<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Distributed inference

Distributed inference splits the workload across multiple GPUs. It a useful technique for fitting larger models in memory and can process multiple prompts for higher throughput.

This guide will show you how to use [Accelerate](https://huggingface.co/docs/accelerate/index) and [PyTorch Distributed](https://pytorch.org/tutorials/beginner/dist_overview.html) for distributed inference.

## Accelerate

Accelerate is a library designed to simplify inference and training on multiple accelerators by handling the setup, allowing users to focus on their PyTorch code.

Install Accelerate with the following command.

```bash
uv pip install accelerate
```

Initialize a [`accelerate.PartialState`] class in a Python file to create a distributed environment. The [`accelerate.PartialState`] class manages process management, device control and distribution, and process coordination.

Move the [`DiffusionPipeline`] to [`accelerate.PartialState.device`] to assign a GPU to each process.

```py
import torch
from accelerate import PartialState
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
    "Qwen/Qwen-Image", torch_dtype=torch.float16
)
distributed_state = PartialState()
pipeline.to(distributed_state.device)
```

Use the [`~accelerate.PartialState.split_between_processes`] utility as a context manager to automatically distribute the prompts between the number of processes.

```py
with distributed_state.split_between_processes(["a dog", "a cat"]) as prompt:
    result = pipeline(prompt).images[0]
    result.save(f"result_{distributed_state.process_index}.png")
```

Call `accelerate launch` to run the script and use the `--num_processes` argument to set the number of GPUs to use.

```bash
accelerate launch run_distributed.py --num_processes=2
```

> [!TIP]
> Refer to this minimal example [script](https://gist.github.com/sayakpaul/cfaebd221820d7b43fae638b4dfa01ba) for running inference across multiple GPUs. To learn more, take a look at the [Distributed Inference with 🤗 Accelerate](https://huggingface.co/docs/accelerate/en/usage_guides/distributed_inference#distributed-inference-with-accelerate) guide.

## PyTorch Distributed

PyTorch [DistributedDataParallel](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) enables [data parallelism](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=data_parallelism), which replicates the same model on each device, to process different batches of data in parallel.

Import `torch.distributed` and `torch.multiprocessing` into a Python file to set up the distributed process group and to spawn the processes for inference on each GPU.

```py
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
    "Qwen/Qwen-Image", torch_dtype=torch.float16,
)
```

Create a function for inference with [init_process_group](https://pytorch.org/docs/stable/distributed.html?highlight=init_process_group#torch.distributed.init_process_group). This method creates a distributed environment with the backend type, the `rank` of the current process, and the `world_size` or number of processes participating (for example, 2 GPUs would be `world_size=2`).

Move the pipeline to `rank` and use `get_rank` to assign a GPU to each process. Each process handles a different prompt.

```py
def run_inference(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    pipeline.to(rank)

    if torch.distributed.get_rank() == 0:
        prompt = "a dog"
    elif torch.distributed.get_rank() == 1:
        prompt = "a cat"

    image = sd(prompt).images[0]
    image.save(f"./{'_'.join(prompt)}.png")
```

Use [mp.spawn](https://pytorch.org/docs/stable/multiprocessing.html#torch.multiprocessing.spawn) to create the number of processes defined in `world_size`.

```py
def main():
    world_size = 2
    mp.spawn(run_inference, args=(world_size,), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
```

Call `torchrun` to run the inference script and use the `--nproc_per_node` argument to set the number of GPUs to use.

```bash
torchrun run_distributed.py --nproc_per_node=2
```

## device_map

The `device_map` argument enables distributed inference by automatically placing model components on separate GPUs. This is especially useful when a model doesn't fit on a single GPU. You can use `device_map` to selectively load and unload the required model components at a given stage as shown in the example below (assumes two GPUs are available).

Set `device_map="balanced"` to evenly distributes the text encoders on all available GPUs. You can use the `max_memory` argument to allocate a maximum amount of memory for each text encoder. Don't load any other pipeline components to avoid memory usage.

```py
from diffusers import FluxPipeline
import torch

prompt = """
cinematic film still of a cat sipping a margarita in a pool in Palm Springs, California
highly detailed, high budget hollywood movie, cinemascope, moody, epic, gorgeous, film grain
"""

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

After the text embeddings are computed, remove them from the GPU to make space for the diffusion transformer.

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

Set `device_map="auto"` to automatically distribute the model on the two GPUs. This strategy places a model on the fastest device first before placing a model on a slower device like a CPU or hard drive if needed. The trade-off of storing model parameters on slower devices is slower inference latency.

```py
from diffusers import AutoModel
import torch 

transformer = AutoModel.from_pretrained(
    "black-forest-labs/FLUX.1-dev", 
    subfolder="transformer",
    device_map="auto",
    torch_dtype=torch.bfloat16
)
```

> [!TIP]
> Run `pipeline.hf_device_map` to see how the various models are distributed across devices. This is useful for tracking model device placement. You can also call `hf_device_map` on the transformer model to see how it is distributed.

Add the transformer model to the pipeline and set the `output_type="latent"` to generate the latents.

```py
pipeline = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
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

Remove the pipeline and transformer from memory and load a VAE to decode the latents. The VAE is typically small enough to be loaded on a single device.

```py
import torch
from diffusers import AutoencoderKL
from diffusers.image_processor import VaeImageProcessor

vae = AutoencoderKL.from_pretrained(ckpt_id, subfolder="vae", torch_dtype=torch.bfloat16).to("cuda")
vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
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

## Context parallelism

[Context parallelism](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=context_parallelism) splits input sequences across multiple GPUs to reduce memory usage. Each GPU processes its own slice of the sequence.

Use [`~ModelMixin.set_attention_backend`] to switch to a more optimized attention backend. Refer to this [table](../optimization/attention_backends#available-backends) for a complete list of available backends.

### Ring Attention

Key (K) and value (V) representations communicate between devices using [Ring Attention](https://huggingface.co/papers/2310.01889). This ensures each split sees every other token's K/V. Each GPU computes attention for its local K/V and passes it to the next GPU in the ring. No single GPU holds the full sequence, which reduces communication latency.

Pass a [`ContextParallelConfig`] to the `parallel_config` argument of the transformer model. The config supports the `ring_degree` argument that determines how many devices to use for Ring Attention.

```py
import torch
from diffusers import AutoModel, QwenImagePipeline, ContextParallelConfig

try:
    torch.distributed.init_process_group("nccl")
    rank = torch.distributed.get_rank()
    device = torch.device("cuda", rank % torch.cuda.device_count())
    torch.cuda.set_device(device)
    
    transformer = AutoModel.from_pretrained("Qwen/Qwen-Image", subfolder="transformer", torch_dtype=torch.bfloat16, parallel_config=ContextParallelConfig(ring_degree=2))
    pipeline = QwenImagePipeline.from_pretrained("Qwen/Qwen-Image", transformer=transformer, torch_dtype=torch.bfloat16, device_map="cuda")
    pipeline.transformer.set_attention_backend("flash")

    prompt = """
    cinematic film still of a cat sipping a margarita in a pool in Palm Springs, California
    highly detailed, high budget hollywood movie, cinemascope, moody, epic, gorgeous, film grain
    """
    
    # Must specify generator so all ranks start with same latents (or pass your own)
    generator = torch.Generator().manual_seed(42)
    image = pipeline(prompt, num_inference_steps=50, generator=generator).images[0]
    
    if rank == 0:
        image.save("output.png")

except Exception as e:
    print(f"An error occurred: {e}")
    torch.distributed.breakpoint()
    raise

finally:
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
```

### Ulysses Attention

[Ulysses Attention](https://huggingface.co/papers/2309.14509) splits a sequence across GPUs and performs an *all-to-all* communication (every device sends/receives data to every other device). Each GPU ends up with all tokens for only a subset of attention heads. Each GPU computes attention locally on all tokens for its head, then performs another all-to-all to regroup results by tokens for the next layer.

[`ContextParallelConfig`] supports Ulysses Attention through the `ulysses_degree` argument. This determines how many devices to use for Ulysses Attention.

Pass the [`ContextParallelConfig`] to [`~ModelMixin.enable_parallelism`].

```py
pipeline.transformer.enable_parallelism(config=ContextParallelConfig(ulysses_degree=2))
```