# xDiT

[xDiT](https://github.com/xdit-project/xDiT) is an inference engine designed for the large scale parallel deployment of Diffusion Transformers (DiTs). xDiT provides a suite of efficient parallel approaches for Diffusion Models, as well as GPU kernel accelerations.

There are four parallel methods supported in xDiT, including [Unified Sequence Parallelism](https://arxiv.org/abs/2405.07719), [PipeFusion](https://arxiv.org/abs/2405.14430), CFG parallelism and data parallelism. The four parallel methods in xDiT can be configured in a hybrid manner, optimizing communication patterns to best suit the underlying network hardware.

Optimization orthogonal to parallelization focuses on accelerating single GPU performance. In addition to utilizing well-known Attention optimization libraries, we leverage compilation acceleration technologies such as torch.compile and onediff.

The overview of xDiT is shown as follows.

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/xDiT/documentation-images/resolve/main/methods/xdit_overview.png">
</div>
You can install xDiT using the following command:


```bash
pip install xfuser
```

Here's an example of using xDiT to accelerate inference of a Diffusers model.

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
    
# do anything you want with pipeline here

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

As you can see, we only need to use xFuserArgs from xDiT to get configuration parameters, and pass these parameters along with the pipeline object from the Diffusers library into xDiTParallel to complete the parallelization of a specific pipeline in Diffusers.

xDiT runtime parameters can be viewed in the command line using `-h`, and you can refer to this [usage](https://github.com/xdit-project/xDiT?tab=readme-ov-file#2-usage) example for more details.

xDiT needs to be launched using torchrun to support its multi-node, multi-GPU parallel capabilities. For example, the following command can be used for 8-GPU parallel inference:

```bash
torchrun --nproc_per_node=8 ./inference.py --model models/FLUX.1-dev --data_parallel_degree 2 --ulysses_degree 2 --ring_degree 2 --prompt "A snowy mountain" "A small dog" --num_inference_steps 50
```

## Supported models

A subset of Diffusers models are supported in xDiT, such as Flux.1, Stable Diffusion 3, etc. The latest supported models can be found [here](https://github.com/xdit-project/xDiT?tab=readme-ov-file#-supported-dits).

## Benchmark
We tested different models on various machines, and here is some of the benchmark data.

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

More detailed performance metric can be found on our [github page](https://github.com/xdit-project/xDiT?tab=readme-ov-file#perf).

## Reference

[xDiT-project](https://github.com/xdit-project/xDiT)

[USP: A Unified Sequence Parallelism Approach for Long Context Generative AI](https://arxiv.org/abs/2405.07719)

[PipeFusion: Displaced Patch Pipeline Parallelism for Inference of Diffusion Transformer Models](https://arxiv.org/abs/2405.14430)