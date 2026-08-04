<!--版权所有 2025 The HuggingFace Team。保留所有权利。

根据 Apache 许可证 2.0 版本（“许可证”）授权；除非遵守许可证，否则不得使用此文件。您可以在以下网址获取许可证副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据许可证分发的软件按“原样”分发，不附带任何明示或暗示的担保或条件。请参阅许可证了解具体的语言管理权限和限制。
-->

# 分布式推理

在分布式设置中，您可以使用 🤗 [Accelerate](https://huggingface.co/docs/accelerate/index) 或 [PyTorch Distributed](https://pytorch.org/tutorials/beginner/dist_overview.html) 在多个 GPU 上运行推理，这对于并行生成多个提示非常有用。

本指南将向您展示如何使用 🤗 Accelerate 和 PyTorch Distributed 进行分布式推理。

## 🤗 Accelerate

🤗 [Accelerate](https://huggingface.co/docs/accelerate/index) 是一个旨在简化在分布式设置中训练或运行推理的库。它简化了设置分布式环境的过程，让您可以专注于您的 PyTorch 代码。

首先，创建一个 Python 文件并初始化一个 [`accelerate.PartialState`] 来创建分布式环境；您的设置会自动检测，因此您无需明确定义 `rank` 或 `world_size`。将 [`DiffusionPipeline`] 移动到 `distributed_state.device` 以为每个进程分配一个 GPU。

现在使用 [`~accelerate.PartialState.split_between_processes`] 实用程序作为上下文管理器，自动在进程数之间分发提示。

```py
import torch
from accelerate import PartialState
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5", dtype=torch.float16, use_safetensors=True
)
distributed_state = PartialState()
pipeline.to(distributed_state.device)

with distributed_state.split_between_processes(["a dog", "a cat"]) as prompt:
    result = pipeline(prompt).images[0]
    result.save(f"result_{distributed_state.process_index}.png")
```

使用 `--num_processes` 参数指定要使用的 GPU 数量，并调用 `accelerate launch` 来运行脚本：

```bash
accelerate launch run_distributed.py --num_processes=2
```

> [!TIP]
> 参考这个最小示例 [脚本](https://gist.github.com/sayakpaul/cfaebd221820d7b43fae638b4dfa01ba) 以在多个 GPU 上运行推理。要了解更多信息，请查看 [使用 🤗 Accelerate 进行分布式推理](https://huggingface.co/docs/accelerate/en/usage_guides/distributed_inference#distributed-inference-with-accelerate) 指南。

## PyTorch Distributed

PyTorch 支持 [`DistributedDataParallel`](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)，它启用了数据
并行性。

首先，创建一个 Python 文件并导入 `torch.distributed` 和 `torch.multiprocessing` 来设置分布式进程组，并为每个 GPU 上的推理生成进程。您还应该初始化一个 [`DiffusionPipeline`]：

```py
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from diffusers import DiffusionPipeline

sd = DiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5", dtype=torch.float16, use_safetensors=True
)
```

您需要创建一个函数来运行推理；[`init_process_group`](https://pytorch.org/docs/stable/distributed.html?highlight=init_process_group#torch.distributed.init_process_group) 处理创建一个分布式环境，指定要使用的后端类型、当前进程的 `rank` 以及参与进程的数量 `world_size`。如果您在 2 个 GPU 上并行运行推理，那么 `world_size` 就是 2。

将 [`DiffusionPipeline`] 移动到 `rank`，并使用 `get_rank` 为每个进程分配一个 GPU，其中每个进程处理不同的提示：

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

要运行分布式推理，调用 [`mp.spawn`](https://pytorch.org/docs/stable/multiprocessing.html#torch.multiprocessing.spawn) 在 `world_size` 定义的 GPU 数量上运行 `run_inference` 函数：

```py
def main():
    world_size = 2
    mp.spawn(run_inference, args=(world_size,), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
```

完成推理脚本后，使用 `--nproc_per_node` 参数指定要使用的 GPU 数量，并调用 `torchrun` 来运行脚本：

```bash
torchrun run_distributed.py --nproc_per_node=2
```

> [!TIP]
> 您可以在 [`DiffusionPipeline`] 中使用 `device_map` 将其模型级组件分布在多个设备上。请参考 [设备放置](../tutorials/inference_with_big_models#device-placement) 指南了解更多信息。

## 模型分片

现代扩散系统，如 [Flux](../api/pipelines/flux)，非常大且包含多个模型。例如，[Flux.1-Dev](https://hf.co/black-forest-labs/FLUX.1-dev) 由两个文本编码器 - [T5-XXL](https://hf.co/google/t5-v1_1-xxl) 和 [CLIP-L](https://hf.co/openai/clip-vit-large-patch14) - 一个 [扩散变换器](../api/models/flux_transformer)，以及一个 [VAE](../api/models/autoencoderkl) 组成。对于如此大的模型，在消费级 GPU 上运行推理可能具有挑战性。

模型分片是一种技术，当模型无法容纳在单个 GPU 上时，将模型分布在多个 GPU 上。下面的示例假设有两个 16GB GPU 可用于推理。

开始使用文本编码器计算文本嵌入。通过设置 `device_map="balanced"` 将文本编码器保持在两个GPU上。`balanced` 策略将模型均匀分布在所有可用GPU上。使用 `max_memory` 参数为每个GPU上的每个文本编码器分配最大内存量。

> [!TIP]
> **仅** 在此步骤加载文本编码器！扩散变换器和VAE在后续步骤中加载以节省内存。

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
    dtype=torch.bfloat16
)
with torch.no_grad():
    print("Encoding prompts.")
    prompt_embeds, pooled_prompt_embeds, text_ids = pipeline.encode_prompt(
        prompt=prompt, prompt_2=None, max_sequence_length=512
    )
```

一旦文本嵌入计算完成，从GPU中移除它们以为扩散变换器腾出空间。

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

接下来加载扩散变换器，它有125亿参数。这次，设置 `device_map="auto"` 以自动将模型分布在两个16GB GPU上。`auto` 策略由 [Accelerate](https://hf.co/docs/accelerate/index) 支持，并作为 [大模型推理](https://hf.co/docs/accelerate/concept_guides/big_model_inference) 功能的一部分可用。它首先将模型分布在最快的设备（GPU）上，然后在需要时移动到较慢的设备如CPU和硬盘。将模型参数存储在较慢设备上的权衡是推理延迟较慢。

```py
from diffusers import AutoModel
import torch 

transformer = AutoModel.from_pretrained(
    "black-forest-labs/FLUX.1-dev", 
    subfolder="transformer",
    device_map="auto",
    dtype=torch.bfloat16
)
```

> [!TIP]
> 在任何时候，您可以尝试 `print(pipeline.hf_device_map)` 来查看各种模型如何在设备上分布。这对于跟踪模型的设备放置很有用。您也可以尝试 `print(transformer.hf_device_map)` 来查看变换器模型如何在设备上分片。

将变换器模型添加到管道中以进行去噪，但将其他模型级组件如文本编码器和VAE设置为 `None`，因为您还不需要它们。

```py
pipeline = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    text_encoder=None,
    text_encoder_2=None,
    tokenizer=None,
    tokenizer_2=None,
    vae=None,
    transformer=transformer,
    dtype=torch.bfloat16
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

从内存中移除管道和变换器，因为它们不再需要。

```py
del pipeline.transformer
del pipeline

flush()
```

最后，使用变分自编码器（VAE）将潜在表示解码为图像。VAE通常足够小，可以在单个GPU上加载。

```py
from diffusers import AutoencoderKL
from diffusers.image_processor import VaeImageProcessor
import torch 

vae = AutoencoderKL.from_pretrained(ckpt_id, subfolder="vae", dtype=torch.bfloat16).to("cuda")
vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)

with torch.no_grad():
    print("运行解码中。")
    latents = FluxPipeline._unpack_latents(latents, height, width, vae_scale_factor)
    latents = (latents / vae.config.scaling_factor) + vae.config.shift_factor

    image = vae.decode(latents, return_dict=False)[0]
    image = image_processor.postprocess(image, output_type="pil")
    image[0].save("split_transformer.png")
```

通过选择性加载和卸载在特定阶段所需的模型，并将最大模型分片到多个GPU上，可以在消费级GPU上运行大型模型的推理。