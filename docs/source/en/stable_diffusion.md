<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

[[open-in-colab]]

# Basic performance

Diffusion is a random process that is computationally demanding. You may need to run the [`DiffusionPipeline`] several times before getting a desired output. That's why it's important to carefully balance generation speed and memory usage in order to iterate faster,

This guide recommends some basic performance tips for using the [`DiffusionPipeline`]. Refer to the Inference Optimization section docs such as [Accelerate inference](./optimization/fp16) or [Reduce memory usage](./optimization/memory) for more detailed performance guides.

## Memory usage

Reducing the amount of memory used indirectly speeds up generation and can help a model fit on device.

The [`~DiffusionPipeline.enable_model_cpu_offload`] method moves a model to the CPU when it is not in use to save GPU memory.

```py
import torch
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
  "stabilityai/stable-diffusion-xl-base-1.0",
  torch_dtype=torch.bfloat16,
  device_map="cuda"
)
pipeline.enable_model_cpu_offload()

prompt = """
cinematic film still of a cat sipping a margarita in a pool in Palm Springs, California
highly detailed, high budget hollywood movie, cinemascope, moody, epic, gorgeous, film grain
"""
pipeline(prompt).images[0]
print(f"Max memory reserved: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
```

## Inference speed

Denoising is the most computationally demanding process during diffusion. Methods that optimizes this process accelerates inference speed. Try the following methods for a speed up.

- Add `device_map="cuda"` to place the pipeline on a GPU. Placing a model on an accelerator, like a GPU, increases speed because it performs computations in parallel.
- Set `torch_dtype=torch.bfloat16` to execute the pipeline in half-precision. Reducing the data type precision increases speed because it takes less time to perform computations in a lower precision.

```py
import torch
import time
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler

pipeline = DiffusionPipeline.from_pretrained(
  "stabilityai/stable-diffusion-xl-base-1.0",
  torch_dtype=torch.bfloat16,
  device_map="cuda
)
```

- Use a faster scheduler, such as [`DPMSolverMultistepScheduler`], which only requires ~20-25 steps.
- Set `num_inference_steps` to a lower value. Reducing the number of inference steps reduces the overall number of computations. However, this can result in lower generation quality.

```py
pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)

prompt = """
cinematic film still of a cat sipping a margarita in a pool in Palm Springs, California
highly detailed, high budget hollywood movie, cinemascope, moody, epic, gorgeous, film grain
"""

start_time = time.perf_counter()
image = pipeline(prompt).images[0]
end_time = time.perf_counter()

print(f"Image generation took {end_time - start_time:.3f} seconds")
```

## Generation quality

Many modern diffusion models deliver high-quality images out-of-the-box. However, you can still improve generation quality by trying the following.

- Try a more detailed and descriptive prompt. Include details such as the image medium, subject, style, and aesthetic. A negative prompt may also help by guiding a model away from undesirable features by using words like low quality or blurry.

    ```py
    import torch
    from diffusers import DiffusionPipeline

    pipeline = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.bfloat16,
        device_map="cuda"
    )

    prompt = """
    cinematic film still of a cat sipping a margarita in a pool in Palm Springs, California
    highly detailed, high budget hollywood movie, cinemascope, moody, epic, gorgeous, film grain
    """
    negative_prompt = "low quality, blurry, ugly, poor details"
    pipeline(prompt, negative_prompt=negative_prompt).images[0]
    ```

    For more details about creating better prompts, take a look at the [Prompt techniques](./using-diffusers/weighted_prompts) doc.

- Try a different scheduler, like [`HeunDiscreteScheduler`] or [`LMSDiscreteScheduler`], that gives up generation speed for quality.

    ```py
    import torch
    from diffusers import DiffusionPipeline, HeunDiscreteScheduler

    pipeline = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.bfloat16,
        device_map="cuda"
    )
    pipeline.scheduler = HeunDiscreteScheduler.from_config(pipeline.scheduler.config)

    prompt = """
    cinematic film still of a cat sipping a margarita in a pool in Palm Springs, California
    highly detailed, high budget hollywood movie, cinemascope, moody, epic, gorgeous, film grain
    """
    negative_prompt = "low quality, blurry, ugly, poor details"
    pipeline(prompt, negative_prompt=negative_prompt).images[0]
    ```

## Next steps

Diffusers offers more advanced and powerful optimizations such as [group-offloading](./optimization/memory#group-offloading) and [regional compilation](./optimization/fp16#regional-compilation). To learn more about how to maximize performance, take a look at the Inference Optimization section.