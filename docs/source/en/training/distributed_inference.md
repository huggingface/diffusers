<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Distributed inference with multiple GPUs

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
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, use_safetensors=True
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

To learn more, take a look at the [Distributed Inference with 🤗 Accelerate](https://huggingface.co/docs/accelerate/en/usage_guides/distributed_inference#distributed-inference-with-accelerate) guide.

</Tip>

### Using `device_map`

<Tip warning={true}>

🧪 This feature is experimental and its APIs might change in future. 

</Tip>

With 🤗 Accelerate, you can make use of a supported `device_map` strategy to distribute the model-level components of a pipeline across multiple devices. The section below explains a situation where performing distributed inference via `device_map` could be useful.

Let's say you have two consumer GPUs each having 8GBs of VRAM. Using `enable_model_cpu_offload()` might not work best here because -

* it works only on a single GPU
* a single model might not fit on a single GPU `enable_sequential_cpu_offload()` might work but will be extremely slow. Here also we're limited to using a single GPU only.

To use the "balanced" `device_map`, just pass `device_map="balanced"` when instantiating the pipeline:

```diff
from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained(
-    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, use_safetensors=True,
+    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, use_safetensors=True, device_map="balanced"
)
image = pipeline("a dog").images[0]
image
```

Pass a dictionary to enforce a restriction on the maximum GPU memory that can be used:

```diff
from diffusers import DiffusionPipeline
import torch

max_memory = {0:"1GB", 1:"1GB"}
pipeline = DiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16, 
    use_safetensors=True, 
    device_map="balanced",
+    max_memory=max_memory
)
image = pipeline("a dog").images[0]
image
```

Reset any `device_map` by calling `reset_device_map()` on the pipeline object.

To be able to use methods like `to()`, `enable_sequential_cpu_offload()`, and `enable_model_cpu_offload()` on a supported pipeline that has been device-mapped, first call `reset_device_map()` on it and then use these methods.

> [!NOTE]  
> Currently, we support only "balanced" `device_map`. We plan to support more device mapping strategies in future.

## PyTorch Distributed

PyTorch supports [`DistributedDataParallel`](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) which enables data parallelism.

To start, create a Python file and import `torch.distributed` and `torch.multiprocessing` to set up the distributed process group and to spawn the processes for inference on each GPU. You should also initialize a [`DiffusionPipeline`]:

```py
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from diffusers import DiffusionPipeline

sd = DiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, use_safetensors=True
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
