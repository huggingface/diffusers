# xDiT

[xDiT](https://github.com/xdit-project/xDiT) 是一个推理引擎，专为大规模并行部署扩散变换器（DiTs）而设计。xDiT 提供了一套用于扩散模型的高效并行方法，以及 GPU 内核加速。

xDiT 支持四种并行方法，包括[统一序列并行](https://huggingface.co/papers/2405.07719)、[PipeFusion](https://huggingface.co/papers/2405.14430)、CFG 并行和数据并行。xDiT 中的这四种并行方法可以以混合方式配置，优化通信模式以最适合底层网络硬件。

与并行化正交的优化侧重于加速单个 GPU 的性能。除了利用知名的注意力优化库外，我们还利用编译加速技术，如 torch.compile 和 onediff。

xDiT 的概述如下所示。

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/xDiT/documentation-images/resolve/main/methods/xdit_overview.png">
</div>
您可以使用以下命令安装 xDiT：

```bash
pip install xfuser
```

以下是一个使用 xDiT 加速 Diffusers 模型推理的示例。

```diff
 import torch
 from diffusers import StableDiffusion3Pipeline

 from xfuser import xFuserArgs, xDiTParallel
 from xfuser.config import FlexibleArgumentParser
 from xfuser.core.distributed import get_world_group

 def main():
+    parser = FlexibleArgumentParser(description="xFuser Arguments")
+    args = xFuserArgs.add_cli_args(parser).parse_args()
+    engine_args = xFuserArgs.from_cli_args(args)
+    engine_config, input_config = engine_args.create_config()

     local_rank = get_world_group().local_rank
     pipe = StableDiffusion3Pipeline.from_pretrained(
         pretrained_model_name_or_path=engine_config.model_config.model,
         torch_dtype=torch.float16,
     ).to(f"cuda:{local_rank}")
    
# 在这里对管道进行任何操作

+    pipe = xDiTParallel(pipe, engine_config, input_config)

     pipe(
         height=input_config.height,
         width=input_config.height,
         prompt=input_config.prompt,
         num_inference_steps=input_config.num_inference_steps,
         output_type=input_config.output_type,
         generator=torch.Generator(device="cuda").manual_seed(input_config.seed),
     )

+    if input_config.output_type == "pil":
+        pipe.save("results", "stable_diffusion_3")

if __name__ == "__main__":
    main()
```

如您所见，我们只需要使用 xDiT 中的 xFuserArgs 来获取配置参数，并将这些参数与来自 Diffusers 库的管道对象一起传递给 xDiTParallel，即可完成对 Diffusers 中特定管道的并行化。

xDiT 运行时参数可以在命令行中使用 `-h` 查看，您可以参考此[使用](https://github.com/xdit-project/xDiT?tab=readme-ov-file#2-usage)示例以获取更多详细信息。
ils。

xDiT 需要使用 torchrun 启动，以支持其多节点、多 GPU 并行能力。例如，以下命令可用于 8-GPU 并行推理：

```bash
torchrun --nproc_per_node=8 ./inference.py --model models/FLUX.1-dev --data_parallel_degree 2 --ulysses_degree 2 --ring_degree 2 --prompt "A snowy mountain" "A small dog" --num_inference_steps 50
```

## 支持的模型

在 xDiT 中支持 Diffusers 模型的一个子集，例如 Flux.1、Stable Diffusion 3 等。最新支持的模型可以在[这里](https://github.com/xdit-project/xDiT?tab=readme-ov-file#-supported-dits)找到。

## 基准测试
我们在不同机器上测试了各种模型，以下是一些基准数据。

### Flux.1-schnell
<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/xDiT/documentation-images/resolve/main/performance/flux/Flux-2k-L40.png">
</div>

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/xDiT/documentation-images/resolve/main/performance/flux/Flux-2K-A100.png">
</div>

### Stable Diffusion 3
<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/xDiT/documentation-images/resolve/main/performance/sd3/L40-SD3.png">
</div>

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/xDiT/documentation-images/resolve/main/performance/sd3/A100-SD3.png">
</div>

### HunyuanDiT
<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/xDiT/documentation-images/resolve/main/performance/hunuyuandit/L40-HunyuanDiT.png">
</div>

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/xDiT/documentation-images/resolve/main/performance/hunuyuandit/V100-HunyuanDiT.png">
</div>

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/xDiT/documentation-images/resolve/main/performance/hunuyuandit/T4-HunyuanDiT.png">
</div>

更详细的性能指标可以在我们的 [GitHub 页面](https://github.com/xdit-project/xDiT?tab=readme-ov-file#perf) 上找到。

## 参考文献

[xDiT-project](https://github.com/xdit-project/xDiT)

[USP: A Unified Sequence Parallelism Approach for Long Context Generative AI](https://huggingface.co/papers/2405.07719)

[PipeFusion: Displaced Patch Pipeline Parallelism for Inference of Diffusion Transformer Models](https://huggingface.co/papers/2405.14430)