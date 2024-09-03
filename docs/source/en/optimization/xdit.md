# xDiT

xDiT is an inference engine designed for the parallel deployment of DiTs on large scale. xDiT provides a suite of efficient parallel approaches for Diffusion Models, as well as GPU kernel accelerations.

<div class="flex justify-center">
    <img src="https://github.com/xdit-project/xDiT/raw/main/assets/methods/xdit_overview.png">
</div>
You can install xDiT using the following command:


```bash
pip install xfuser
```

Here's an example of using xDiT to accelerate the inference of a diffusers model:

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

As you can see, we only need to use xFuserArgs from xDiT to get configuration parameters, and pass these parameters along with the pipeline object from the diffusers library into xDiTParallel to complete the parallelization of a specific pipeline in diffusers.


xDiT runtime parameters can be viewed in the command line using -h, and detailed introductions can also be found on the [xDiT Github page](https://github.com/xdit-project/xDiT?tab=readme-ov-file#2-usage).


xDiT needs to be launched using torchrun to support its multi-node, multi-GPU parallel capabilities. For example, the following command can be used for 8-GPU parallel inference:

```bash
torchrun --nproc_per_node=8 ./inference.py --model models/FLUX.1-dev --data_parallel_degree 2 --ulysses_degree 2 --ring_degree 2 --prompt "A snowy mountain" "A small dog" --num_inference_steps 50
```

# Supported Models

We have supported a subset of diffusers models in xDiT, including the most popular models such as Flux.1, Stable Diffusion 3, etc. The latest supported models can be found on https://github.com/xdit-project/xDiT?tab=readme-ov-file#-supported-dits

# Benchmark

We tested different models on various machines. Here is some of the data:


## Flux.1-schnell
<div class="flex justify-center">
    <img src="https://github.com/xdit-project/xDiT/raw/main/assets/performance/flux/Flux-2k-L40.png">
    <img src="https://github.com/xdit-project/xDiT/raw/main/assets/performance/flux/Flux-2K-A100.png">
</div>

## Stable Diffusion 3
<div class="flex justify-center">
    <img src="https://github.com/xdit-project/xDiT/raw/main/assets/performance/sd3/L40-SD3.png">
    <img src="https://github.com/xdit-project/xDiT/raw/main/assets/performance/sd3/A100-SD3.png">
</div>

## HunyuanDiT
<div class="flex justify-center">
    <img src="https://github.com/xdit-project/xDiT/raw/main/assets/performance/hunuyuandit/L40-HunyuanDiT.png">
    <img src="https://github.com/xdit-project/xDiT/raw/main/assets/performance/hunuyuandit/A100-HunyuanDiT.png">
    <img src="https://github.com/xdit-project/xDiT/raw/main/assets/performance/hunuyuandit/T4-HunyuanDiT.png">
</div>

More detailed performance metric can be found on our [github page](https://github.com/xdit-project/xDiT?tab=readme-ov-file#perf).

# Reference

[xDiT-project](https://github.com/xdit-project/xDiT)

[USP: A Unified Sequence Parallelism Approach for Long Context Generative AI](https://arxiv.org/abs/2405.07719)

[PipeFusion: Displaced Patch Pipeline Parallelism for Inference of Diffusion Transformer Models](https://arxiv.org/abs/2405.14430)