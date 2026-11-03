<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Data parallelism

[Data parallelism](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=data_parallelism) replicates the same model on each GPU and gives each GPU different prompts. It increases throughput when the model already fits on one GPU, and it leaves the latency of a single prompt unchanged. If you haven't already, try [Batch inference](../using-diffusers/batched_inference) on one GPU first. Refer to [Distributed inference](../training/distributed_inference) to compare it with the other strategies.

You can set up data parallelism with [Accelerate](https://huggingface.co/docs/accelerate/index) or with [PyTorch Distributed](https://pytorch.org/tutorials/beginner/dist_overview.html) directly.

## Accelerate

Accelerate sets up the processes and assigns a GPU to each one.

Install Accelerate with the following command.

```bash
uv pip install accelerate
```

Create a [`~accelerate.PartialState`] in a file called `run_distributed.py` to set up the distributed environment. Move the [`DiffusionPipeline`] to the `PartialState.device` to assign a GPU to each process.

```py
import torch
from accelerate import PartialState
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
    "Qwen/Qwen-Image", dtype=torch.bfloat16
)
distributed_state = PartialState()
pipeline.to(distributed_state.device)
```

In the same file, use [`~accelerate.PartialState.split_between_processes`] as a context manager to split the prompts between processes.

```py
with distributed_state.split_between_processes(["a dog", "a cat"]) as prompt:
    result = pipeline(prompt).images[0]
    result.save(f"result_{distributed_state.process_index}.png")
```

Run the script with `accelerate launch`, and set `--num_processes` to the number of GPUs.

```bash
accelerate launch --num_processes=2 run_distributed.py
```

> [!TIP]
> For a complete example, see this [minimal multi-GPU inference script](https://gist.github.com/sayakpaul/cfaebd221820d7b43fae638b4dfa01ba). To learn more, see the [Distributed inference with Accelerate](https://huggingface.co/docs/accelerate/en/usage_guides/distributed_inference#distributed-inference-with-accelerate) guide.

## PyTorch Distributed

You can also set up data parallelism with `torch.distributed` directly, without Accelerate.

Call [init_process_group](https://pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group) to create the distributed environment. `torchrun` sets the rank of each process and the total number of processes (the world size) through environment variables, so you don't pass them yourself.

Use `get_rank` to move the pipeline to a different GPU in each process and to give each process a different prompt.

```py
import torch
import torch.distributed as dist
from diffusers import DiffusionPipeline

def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = torch.device(f"cuda:{rank}")

    pipeline = DiffusionPipeline.from_pretrained(
        "Qwen/Qwen-Image", dtype=torch.bfloat16,
    ).to(device)

    prompts = ["a dog", "a cat"]
    prompt = prompts[rank]

    image = pipeline(prompt).images[0]
    image.save(f"{prompt.replace(' ', '_')}.png")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
```

Save the script as `run_distributed.py` and run it with `torchrun`. Set `--nproc-per-node` to the number of GPUs. The `prompts` list has one prompt per process, so add a prompt for each GPU you use.

```bash
torchrun --nproc-per-node 2 run_distributed.py
```
