<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Control image brightness

The Stable Diffusion pipeline is mediocre at generating images that are either very bright or dark as explained in the [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://huggingface.co/papers/2305.08891) paper. The solutions proposed in the paper are currently implemented in the [`DDIMScheduler`] which you can use to improve the lighting in your images.

<Tip>

💡 Take a look at the paper linked above for more details about the proposed solutions!

</Tip>

One of the solutions is to train a model with *v prediction* and *v loss*. Add the following flag to the [`train_text_to_image.py`](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py) or [`train_text_to_image_lora.py`](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_lora.py) scripts to enable `v_prediction`:

```bash
--prediction_type="v_prediction"
```

For example, let's use the [`ptx0/pseudo-journey-v2`](https://huggingface.co/ptx0/pseudo-journey-v2) checkpoint which has been finetuned with `v_prediction`.

Next, configure the following parameters in the [`DDIMScheduler`]:

1. `rescale_betas_zero_snr=True`, rescales the noise schedule to zero terminal signal-to-noise ratio (SNR)
2. `timestep_spacing="trailing"`, starts sampling from the last timestep

```py
from diffusers import DiffusionPipeline, DDIMScheduler

pipeline = DiffusionPipeline.from_pretrained("ptx0/pseudo-journey-v2", use_safetensors=True)

# switch the scheduler in the pipeline to use the DDIMScheduler
pipeline.scheduler = DDIMScheduler.from_config(
    pipeline.scheduler.config, rescale_betas_zero_snr=True, timestep_spacing="trailing"
)
pipeline.to("cuda")
```

Finally, in your call to the pipeline, set `guidance_rescale` to prevent overexposure:

```py
prompt = "A lion in galaxies, spirals, nebulae, stars, smoke, iridescent, intricate detail, octane render, 8k"
image = pipeline(prompt, guidance_rescale=0.7).images[0]
image
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/zero_snr.png"/>
</div>
