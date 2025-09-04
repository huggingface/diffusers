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

## Resources

- Take a look at this [script](https://gist.github.com/sayakpaul/cfaebd221820d7b43fae638b4dfa01ba) for a minimal example of distributed inference with Accelerate.
- For more details, check out Accelerate's [Distributed inference](https://huggingface.co/docs/accelerate/en/usage_guides/distributed_inference#distributed-inference-with-accelerate) guide.
- The `device_map` argument assign models or an entire pipeline to devices. Refer to the [device placement](../using-diffusers/loading#device-placement) docs for more information.