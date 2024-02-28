<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Community tips and tricks

Diffusers owes much of its success to its community of users and contributors. ❤️ This guide is a collection of tips and tricks for using Diffusers shared by community members. It includes helpful advice such as how to customize and implement specific features through callbacks and how to generate high-quality images.

If you have a tip or trick you'd like to share, we'd love to [hear from you](https://github.com/huggingface/diffusers/issues/new/choose)!

## Callback to display image after each generation step

> [!TIP]
> This tip was contributed by [asomoza](https://github.com/asomoza).

Display an image after each generation step by using a [callback](../using-diffusers/callback) to access and manipulate the latents after each step and convert them into an image.

1. Use the function below to convert the SDXL latents (4 channels) to RGB tensors (3 channels) as explained in the [Explaining the SDXL latent space](https://huggingface.co/blog/TimothyAlexisVass/explaining-the-sdxl-latent-space) blog post:

```py
def latents_to_rgb(latents):
    weights = (
        (60, -60, 25, -70),
        (60,  -5, 15, -50),
        (60,  10, -5, -35)
    )

    weights_tensor = torch.t(torch.tensor(weights, dtype=latents.dtype).to(latents.device))
    biases_tensor = torch.tensor((150, 140, 130), dtype=latents.dtype).to(latents.device)
    rgb_tensor = torch.einsum("...lxy,lr -> ...rxy", latents, weights_tensor) + biases_tensor.unsqueeze(-1).unsqueeze(-1)
    image_array = rgb_tensor.clamp(0, 255)[0].byte().cpu().numpy()
    image_array = image_array.transpose(1, 2, 0)

    return Image.fromarray(image_array)
```

2. Create a function to decode and save the latents into an image.

```py
def decode_tensors(pipe, step, timestep, callback_kwargs):
    latents = callback_kwargs["latents"]
    
    image = latents_to_rgb(latents)
    image.save(f"{step}.png")

    return callback_kwargs
```

3. Pass the `decode_tensors` function to the `callback_on_step_end` parameter to decode the tensors after each step. You need to also specify what you want to modify in the `callback_on_step_end_tensor_inputs` parameter, which in this case are the latents.

```py
from diffusers import AutoPipelineForText2Image
import torch
from PIL import Image

pipeline = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
).to("cuda")

image = pipe(
    prompt = "A croissant shaped like a cute bear."
    negative_prompt = "Deformed, ugly, bad anatomy"
    callback_on_step_end=decode_tensors,
    callback_on_step_end_tensor_inputs=["latents"],
).images[0]
```

> [!TIP]
> The latent space is compressed to 128x128 so the images are also 128x128 which is useful for a quick preview.

<div class="flex gap-4">
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/tips_step_0.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">step 0</figcaption>
  </div>
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/tips_step_19.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">step 19
    </figcaption>
  </div>
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/tips_step_29.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">step 29</figcaption>
  </div>
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/tips_step_39.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">step 39</figcaption>
  </div>
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/tips_step_49.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">step 49</figcaption>
  </div>
</div>

## High quality anime images

> [!TIP]
> This tip was contributed by [asomoza](https://github.com/asomoza).

Generating high-quality anime images is a popular application of diffusion models. To achieve this in Diffusers:

1. Choose a good anime model like [Counterfeit](https://hf.co/gsdf/Counterfeit-V3.0) and pair it with negative prompt embeddings such as EasyNegative to further improve the quality of the generated images.

```py
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch

pipeline = StableDiffusionPipeline.from_single_file(
    "https://huggingface.co/gsdf/Counterfeit-V3.0/blob/main/Counterfeit-V3.0_fix_fp16.safetensors",
    torch_dtype=torch.float16,
)
pipeline.load_textual_inversion(
    "embed/EasyNegative",
    weight_name="EasyNegative.safetensors",
    token="EasyNegative"
)
```

2. This is optional, but if there is a specific style (typically a LoRA adapter) you want to apply to the images, download the weights and use the [`load_lora_weights`] method to add it to the pipeline. This example uses the [Dungeon Meshi Marcille Character Lora](https://civitai.com/models/106199/dungeon-meshi-marcille-character-lora).

```py
!wget https://civitai.com/api/download/models/114049 -O marcille.safetensors
pipeline.load_lora_weights('.', weight_name="marcille.safetensors")
```

3. Load a scheduler and set `use_karras_sigmas=True` to use the DPM++ 2M Karras scheduler (take a look at this [scheduler table](../api/schedulers/overview.) to find the A1111 equivalent scheduler in Diffusers).

```py
pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
pipeline.scheduler.config.use_karras_sigmas=True
pipeline.to('cuda')
```

4. Create your prompt and negative prompts, and remember to use the trigger words for this specific LoRA adapter (`dmarci`) and embeddings (`EasyNegative`). It is also important to set the:

    - `lora_scale` parameter to control how to scale the output with the LoRA weights.
    - `clip_skip` parameter to specify the layers of the CLIP model to use. This parameter is especially important for anime checkpoints because it controls how closely aligned the text prompt and image are. A higher `clip_skip` value produces more abstract images.

```py
generator = torch.Generator("cpu").manual_seed(0)

prompt = "dmarci, masterpiece, best quality, 1girl, solo, marcillessa, red choker, detailed and beautiful eyes, (cowboy shot:1.2), HAPPY, walking, jumping,(Turtleneck_sweater:1.4), (Leather_skirt:1.3)"
negative_prompt = "EasyNegative, (worst quality, low quality, bad quality, normal quality:2), logo, text, blurry, low quality, bad anatomy, lowres, normal quality, monochrome, grayscale, worstquality, signature, watermark, cropped, bad proportions, out of focus, username, bad body, long body, (fat:1.2), long neck, deformed, mutated, mutation, ugly, disfigured, poorly drawn face, skin blemishes, skin spots, acnes, missing limb, malformed limbs, floating limbs, disconnected limbs, extra limb, extra arms, mutated hands, poorly drawn hands, malformed hands, mutated hands and fingers, bad hands, missing fingers, fused fingers, too many fingers, extra legs, bad feet, backlighting"

lora_scale = 1.0
images = pipeline(prompt, width=768, height=768, negative_prompt=negative_prompt, num_inference_steps=20, cross_attention_kwargs={"scale": lora_scale}, generator=generator, num_images_per_prompt=4, clip_skip=2, guidance_scale=7).images[0]
images
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/tips_anime_models.png">
</div>

## Increase image details with negative noise

> [!TIP]
> This tip was contributed by [asomoza](https://github.com/asomoza).

Negative noise can increase the level of details in the generated image because it allows the model more "creative freedom". You can pass a noisy image created from the original image or a noise algorithms to the model.