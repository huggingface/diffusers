<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Switch loaded pipelines

[[open-in-colab]]

There are many diffuser pipelines that use the same pre-trained model as [`StableDiffusionPipeline`] and [`StableDiffusionXLPipeline`], but they implement specific features to help you achieve better generation results. This guide will show you how to use the `from_pipe` API to create multiple pipelines without increasing memory usage. By using this approach, you can easily switch between pipelines to use different features.

Let's take an example where we first create a [`StableDiffusionPipeline`] and then reuse the already loaded model components to create a [`StableDiffusionSAGPipeline`] to enhance generation quality.

we will generate an image of a bear eating pizza using Stable Diffusion with the IP-Adapter

```python
from diffusers import DiffusionPipeline, StableDiffusionSAGPipeline
import torch
import gc
from diffusers.utils import load_image
from accelerate.utils import compute_module_sizes

base_repo = "SG161222/Realistic_Vision_V6.0_B1_noVAE"
num_inference_steps = 50
image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/load_neg_embed.png")
prompt="bear eats pizza"
negative_prompt = "wrong white balance, dark, sketches,worst quality,low quality"

pipe_sd = DiffusionPipeline.from_pretrained(base_repo, torch_dtype=torch.float16)
pipe_sd.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
pipe_sd.set_ip_adapter_scale(0.6)
pipe_sd.to("cuda")

generator = torch.Generator(device="cpu").manual_seed(33)
out_sd = pipe_sd(
    prompt=prompt,
    negative_prompt=negative_prompt, 
    ip_adapter_image=image,
    num_inference_steps=num_inference_steps,
    generator=generator,
).images[0]
```

let’s take a look at the image and also print out the memory used 

<div class="flex justify-center">
  <img class="rounded-xl" src="https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/from_pipe_out_sd_0.png"/>
</div>

```python
def bytes_to_giga_bytes(bytes):
    return bytes / 1024 / 1024 / 1024
print(
    f"Max memory allocated: {bytes_to_giga_bytes(torch.cuda.max_memory_allocated())} GB"
)
```

```bash
Max memory allocated: 4.406213283538818 GB
```

Now, we can use `from_pipe` to switch to the SAG pipeline. 

```python
pipe_sag = StableDiffusionSAGPipeline.from_pipe(
    pipe_sd,
)
```

It already has IP-Adapter loaded so that you can pass the same bear image as `ip_adapter_image`

```python
generator = torch.Generator(device="cpu").manual_seed(33)
out_sag = pipe_sag(
    prompt = prompt, 
    negative_prompt=negative_prompt, 
    ip_adapter_image=image,
    num_inference_steps=num_inference_steps,
    generator=generator,
    guidance_scale=1.0,
    sag_scale=0.75).images[0]
```

You can see a pretty nice improvement in the output

<div class="flex justify-center">
  <img class="rounded-xl" src="https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/from_pipe_out_sag_1.png"/>
</div>

Now we have both `stableDiffusionPipeline` and `StableDiffusionSAGPipeline` co-existing with the same loaded model components;  You can use them interchangeably without additional memory.

```
print(
    f"Max memory allocated: {bytes_to_giga_bytes(torch.cuda.max_memory_allocated())} GB"
)
```

```bash
Max memory allocated: 4.406213283538818 GB
```

Let's unload the IP adapter from the SAG pipeline. It's important to note that methods like `load_ip_adapter` and `unload_ip_adapter` modify the state of the model components. Therefore, when you use these methods on one pipeline, it will affect all other pipelines that share the same model components.

```bash
pipe_sag.unload_ip_adapter()
```

If you try to use the Stable Diffusion pipeline with IP adapter again, it will fail

```bash
generator = torch.Generator(device="cpu").manual_seed(33)
out_sd = pipe_sd(
    prompt=prompt,
    negative_prompt=negative_prompt, 
    ip_adapter_image=image,
    num_inference_steps=num_inference_steps,
    generator=generator,
).images[0]
```

```bash
AttributeError: 'NoneType' object has no attribute 'image_projection_layers'
```

Please note that the pipeline methods may not function properly on a new pipeline created using the `from_pipe` method. For instance, the `enable_model_cpu_offload` method installs hooks to the model components based on a unique offloading sequence for each pipeline. Therefore, if the models are executed in a different order in the new pipeline, the CPU offloading may not work correctly.

To ensure proper functionality, we recommend re-applying the pipeline methods on the new pipeline created using the `from_pipe` method.

You can also add or subtract model components when you create new pipelines. Let's now create a AnimateDiff pipeline with an additional `MotionAdapter` module

```bash
from diffusers import AnimateDiffPipeline, MotionAdapter, DDIMScheduler
from diffusers.utils import export_to_gif

adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2", torch_dtype=torch.float16)

pipe_animate = AnimateDiffPipeline.from_pipe(pipe_sd, motion_adapter=adapter)
pipe_animate.scheduler = DDIMScheduler.from_config(pipe_animate.scheduler.config, beta_schedule="linear")
# load ip_adapter again and load lora weights
pipe_animate.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
pipe_animate.load_lora_weights("guoyww/animatediff-motion-lora-zoom-out", adapter_name="zoom-out")
pipe_animate.to("cuda")

generator = torch.Generator(device="cpu").manual_seed(33)
pipe_animate.set_adapters("zoom-out", adapter_weights=0.75)
out = pipe_animate(
    prompt= prompt,
    num_frames=16,
    num_inference_steps=num_inference_steps,
    ip_adapter_image = image,
    generator=generator,
).frames[0]
export_to_gif(out, "out_animate.gif")
```
<div class="flex justify-center">
  <img class="rounded-xl" src="https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/from_pipe_out_animate_3.gif"/>
</div>


When creating multiple pipelines using the `from_pipe` method, it is important to note that the memory requirement will be determined by the pipeline with the highest memory usage. This means that regardless of the number of pipelines you create, the total memory requirement will always be the same as the highest memory requirement among the pipelines.

For example, we have created three pipelines - `stableDiffusionPipeline`, `StableDiffusionSAGPipeline`, and `AnimateDiffPipeline` - and the `AnimateDiffPipeline` has the highest memory requirement, then the total memory usage will be based on the memory requirement of the `AnimateDiffPipeline`. 

Therefore, creating additional pipelines will not add up to the total memory requirement. Each pipeline can be used interchangeably without any additional memory overhead.


Did you know that you can use `from_pipe` with a community pipeline? Let me show you an example of using long negative prompt and prompt weighting!

```bash
pipe_lpw = DiffusionPipeline.from_pipe(
    pipe_sd,
    custom_pipeline="lpw_stable_diffusion",
).to("cuda")

prompt = "best_quality (1girl:1.3) bow bride brown_hair closed_mouth frilled_bow frilled_hair_tubes frills (full_body:1.3) fox_ear hair_bow hair_tubes happy hood japanese_clothes kimono long_sleeves red_bow smile solo tabi uchikake white_kimono wide_sleeves cherry_blossoms"
neg_prompt = "lowres, bad_anatomy, error_body, error_hair, error_arm, error_hands, bad_hands, error_fingers, bad_fingers, missing_fingers, error_legs, bad_legs, multiple_legs, missing_legs, error_lighting, error_shadow, error_reflection, text, error, extra_digit, fewer_digits, cropped, worst_quality, low_quality, normal_quality, jpeg_artifacts, signature, watermark, username, blurry"
generator = torch.Generator(device="cpu").manual_seed(33)
out_lpw = pipe_lpw.text2img(
    prompt, 
    negative_prompt=neg_prompt, 
    width=512,height=512,
    max_embeddings_multiples=3, 
    num_inference_steps=num_inference_steps,
    generator=generator,
    ).images[0]
```

<div class="flex justify-center">
  <img class="rounded-xl" src="https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/from_pipe_out_lpw_4.png"/>
</div>

let’s run StableDiffusionPipeline with the same inputs to compare:  the result from the long prompt weighting pipeline is more aligned with the text prompt.

```
generator = torch.Generator(device="cpu").manual_seed(33)
out_sd = pipe_sd(
    prompt=prompt,
    negative_prompt=negative_prompt,
    generator=generator,
    num_inference_steps=num_inference_steps,
).images[0]
out_sd
```
<div class="flex justify-center">
  <img class="rounded-xl" src="https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/from_pipe_out_sd_5.png"/>
</div>


You can easily switch between different pipelines using the `from_pipe` method, similar to turning on and off a feature on your pipeline. To switch between tasks, you can use the `from_pipe` method with `AutoPipeline`, which automatically identifies the pipeline class based on the task. You can find more information about this feature at the [AutoPipe Guide](https://huggingface.co/docs/diffusers/tutorials/autopipeline).
