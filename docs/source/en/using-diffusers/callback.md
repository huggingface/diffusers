<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Pipeline callbacks

The denoising loop of a pipeline can be modified with custom defined functions using the `callback_on_step_end` parameter. This can be really useful for *dynamically* adjusting certain pipeline attributes, or modifying tensor variables. The flexibility of callbacks opens up some interesting use-cases such as changing the prompt embeddings at each timestep, assigning different weights to the prompt embeddings, and editing the guidance scale.

This guide will show you how to use the `callback_on_step_end` parameter to disable classifier-free guidance (CFG) after 40% of the inference steps to save compute with minimal cost to performance.

The callback function should have the following arguments:

* `pipe` (or the pipeline instance) provides access to useful properties such as `num_timestep` and `guidance_scale`. You can modify these properties by updating the underlying attributes. For this example, you'll disable CFG by setting `pipe._guidance_scale=0.0`.
* `step_index` and `timestep` tell you where you are in the denoising loop. Use `step_index` to turn off CFG after reaching 40% of `num_timestep`.
* `callback_kwargs` is a dict that contains tensor variables you can modify during the denoising loop. It only includes variables specified in the `callback_on_step_end_tensor_inputs` argument, which is passed to the pipeline's `__call__` method. Different pipelines may use different sets of variables, so please check a pipeline's `_callback_tensor_inputs` attribute for the list of variables you can modify. Some common variables include `latents` and `prompt_embeds`. For this function, change the batch size of `prompt_embeds` after setting `guidance_scale=0.0` in order for it to work properly.

Your callback function should look something like this:

```python
def callback_dynamic_cfg(pipe, step_index, timestep, callback_kwargs):
        # adjust the batch_size of prompt_embeds according to guidance_scale
        if step_index == int(pipe.num_timestep * 0.4):
                prompt_embeds = callback_kwargs["prompt_embeds"]
                prompt_embeds = prompt_embeds.chunk(2)[-1]

        # update guidance_scale and prompt_embeds
        pipe._guidance_scale = 0.0
        callback_kwargs["prompt_embeds"] = prompt_embeds
        return callback_kwargs
```

Now, you can pass the callback function to the `callback_on_step_end` parameter and the `prompt_embeds` to `callback_on_step_end_tensor_inputs`.

```py
import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt = "a photo of an astronaut riding a horse on mars"

generator = torch.Generator(device="cuda").manual_seed(1)
out = pipe(prompt, generator=generator, callback_on_step_end=callback_custom_cfg, callback_on_step_end_tensor_inputs=['prompt_embeds'])

out.images[0].save("out_custom_cfg.png")
```

The callback function is executed at the end of each denoising step, and modifies the pipeline attributes and tensor variables for the next denoising step.

With callbacks, you can implement features such as dynamic CFG without having to modify the underlying code at all!

<Tip>

🤗 Diffusers currently only supports `callback_on_step_end`, but feel free to open a [feature request](https://github.com/huggingface/diffusers/issues/new/choose) if you have a cool use-case and require a callback function with a different execution point!

</Tip>
