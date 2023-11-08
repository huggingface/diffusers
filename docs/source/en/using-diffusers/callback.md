<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Using callback 

[[open-in-colab]]

Most 🤗 Diffusers pipeline now accept a `callback_on_step_end` argument that allows you to change the default behavior of denoising loop with custom defined functions. Here is an example of a callback function we can write to disable classifier free guidance after 40% of inference steps to save compute with minimum tradeoff in performance.

```python
def callback_dynamic_cfg(pipe, step_index, timestep, callback_kwargs):    
        # adjust the batch_size of prompt_embeds according to guidance_scale
        if step_index == int(pipe.num_timestep * 0.4):
                prompt_embeds = callback_kwargs["prompt_embeds"]
                prompt_embeds =prompt_embeds.chunk(2)[-1]

        # update guidance_scale and prompt_embeds
        pipe._guidance_scale = 0.0
        callback_kwargs["prompt_embeds"] = prompt_embeds
        return callback_kwargs
```

Your callback function has below arguments:
* `pipe` is the pipeline instance, which provides access to useful properties such as `num_timestep` and `guidance_scale`. You can modify these properties by updating the underlying attributes. In this example, we disable CFG by setting `pipe._guidance_scale` to be `0`.
* `step_index` and `timestep` tell you where you are in the denoising loop. In our example, we use `step_index` to decide when to turn off CFG.
* `callback_kwargs` is a dict that contains tensor variables you can modify during the denoising loop. It only includes variables specified in the `callback_on_step_end_tensor_inputs` argument passed to the pipeline's `__call__` method. Different pipelines may use different sets of variables so please check the pipeline class's `_callback_tensor_inputs` attribute for the list of variables that you can modify. Common variables include `latents` and `prompt_embeds`. In our example, we need to adjust the batch size of `prompt_embeds` after setting `guidance_scale` to be `0` in order for it to work properly.

You can pass the callback function as `callback_on_step_end` argument to the pipeline along with `callback_on_step_end_tensor_inputs`. 

```
import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt = "a photo of an astronaut riding a horse on mars"

generator = torch.Generator(device="cuda").manual_seed(1)
out= pipe(prompt, generator=generator, callback_on_step_end = callback_custom_cfg, callback_on_step_end_tensor_inputs=['prompt_embeds'])

out.images[0].save("out_custom_cfg.png")
```

Your callback function will be executed at the end of each denoising step and modify pipeline attributes and tensor variables for the next denoising step. We successfully added the "dynamic CFG" feature to the stable diffusion pipeline without having to modify the code at all.

<Tip>

Currently we only support `callback_on_step_end`. If you have a solid use case and require a callback function with a different execution point, please open an [feature request](https://github.com/huggingface/diffusers/issues/new/choose) so we can add it!

</Tip>