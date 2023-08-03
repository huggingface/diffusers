<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Weighting prompts

[[open-in-colab]]

Text-guided diffusion models generate images based on a given text prompt. The text prompt
can include multiple concepts that the model should generate and it's often desirable to weight
certain parts of the prompt more or less. 

Diffusion models work by conditioning the cross attention layers of the diffusion model with contextualized text embeddings (see the [Stable Diffusion Guide for more information](../stable-diffusion)).
Thus a simple way to emphasize (or de-emphasize) certain parts of the prompt is by increasing or reducing the scale of the text embedding vector that corresponds to the relevant part of the prompt.
This is called "prompt-weighting" and has been a highly demanded feature by the community (see issue [here](https://github.com/huggingface/diffusers/issues/2431)).

## How to do prompt-weighting in Diffusers

We believe the role of `diffusers` is to be a toolbox that provides essential features that enable other projects, such as [InvokeAI](https://github.com/invoke-ai/InvokeAI) or [diffuzers](https://github.com/abhishekkrthakur/diffuzers), to build powerful UIs. In order to support arbitrary methods to manipulate prompts, `diffusers` exposes a [`prompt_embeds`](https://huggingface.co/docs/diffusers/en/api/pipelines/stable_diffusion/text2img#diffusers.StableDiffusionPipeline.__call__.prompt_embeds) function argument and an optional [`negative_prompt_embeds`](https://huggingface.co/docs/diffusers/en/api/pipelines/stable_diffusion/text2img#diffusers.StableDiffusionPipeline.__call__.negative_prompt_embeds) function argument to many pipelines such as [`StableDiffusionPipeline`], [`StableDiffusionControlNetPipeline`], [`StableDiffusionXLPipeline`], allowing to directly pass the "prompt-weighted"/scaled text embeddings to the pipeline.

The [compel library](https://github.com/damian0815/compel) provides an easy way to emphasize or de-emphasize portions of the prompt for you. We strongly recommend it instead of preparing the embeddings yourself.

Let's look at a simple example. Imagine you want to generate an image of `"a red cat playing with a ball"` as 
follows:


### StableDiffusionPipeline

```py
from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

prompt = "a red cat playing with a ball"

generator = torch.Generator(device="cpu").manual_seed(33)

image = pipe(prompt, generator=generator, num_inference_steps=20).images[0]
image
```

This gives you:

![img](https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/compel/forest_0.png)

As you can see, there is no "ball" in the image. Let's emphasize this part!

For this we should install the `compel` library:

```py
pip install compel --upgrade
```

and then create a `Compel` object:

```py
from compel import Compel

compel_proc = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)
```

Now we emphasize the part "ball" with the `"++"` syntax:

```py
prompt = "a red cat playing with a ball++"
```

and instead of passing this to the pipeline directly, we have to process it using `compel_proc`:

```py
prompt_embeds = compel_proc(prompt)
```

Now we can pass `prompt_embeds` directly to the pipeline:

```py
generator = torch.Generator(device="cpu").manual_seed(33)

images = pipe(prompt_embeds=prompt_embeds, generator=generator, num_inference_steps=20).images[0]
image
```

We now get the following image which has a "ball"!

![img](https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/compel/forest_1.png)

Similarly, we de-emphasize parts of the sentence by using the `--` suffix for words, feel free to give it 
a try!

If your favorite pipeline does not have a `prompt_embeds` input, please make sure to open an issue, the 
diffusers team tries to be as responsive as possible.

Compel 1.1.6 adds a utility class to simplify using textual inversions.  Instantiate a `DiffusersTextualInversionManager` and pass it to Compel init:

```
textual_inversion_manager = DiffusersTextualInversionManager(pipe)
compel = Compel(
    tokenizer=pipe.tokenizer,
    text_encoder=pipe.text_encoder,
    textual_inversion_manager=textual_inversion_manager)
```

Also, please check out the documentation of the [compel](https://github.com/damian0815/compel) library for 
more information.

### StableDiffusionXLPipeline

For StableDiffusionXL we need to not only pass `prompt_embeds` (and optionally `negative_prompt_embeds`), but also [`pooled_prompt_embeds`](https://huggingface.co/docs/diffusers/en/api/pipelines/stable_diffusion/stable_diffusion_xl#diffusers.StableDiffusionXLInpaintPipeline.__call__.pooled_prompt_embeds) and optionally [`negative_pooled_prompt_embeds`](https://huggingface.co/docs/diffusers/en/api/pipelines/stable_diffusion/stable_diffusion_xl#diffusers.StableDiffusionXLInpaintPipeline.__call__.negative_pooled_prompt_embeds). 
In addition, [`StableDiffusionXLPipeline`] has two tokenizers and two text encoders which both need to be used to weight the prompt.
Luckily, [`compel`](https://github.com/damian0815/compel) takes care of SDXL's special needs - all we have to do is to pass both tokenizers and text encoders to the `Compel` class.


```py
from compel import Compel, ReturnedEmbeddingsType
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
  "stabilityai/stable-diffusion-xl-base-1.0",
  variant="fp16",
  use_safetensors=True,
  torch_dtype=torch.float16
).to("cuda")

compel = Compel(
  tokenizer=[pipeline.tokenizer, pipeline.tokenizer_2] ,
  text_encoder=[pipeline.text_encoder, pipeline.text_encoder_2],
  returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
  requires_pooled=[False, True]
)
```

Let's try our example from above again. We use the same seed for both prompts and upweight ball by a factor of 1.5 for the first 
prompt and downweight ball by 40% for the second prompt.

```py
# upweight "ball"
prompt = ["a red cat playing with a (ball)1.5", "a red cat playing with a (ball)0.6"]
conditioning, pooled = compel(prompt)


# generate image
generator = [torch.Generator().manual_seed(33) for _ in range(len(prompt))]
images = pipeline(prompt_embeds=conditioning, pooled_prompt_embeds=pooled, generator=generator, num_inference_steps=30).images
```

Let's have a look at the result.

<div class="flex gap-4">
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/compel/sdxl_ball1.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">"a red cat playing with a (ball)1.5"</figcaption>
  </div>
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/compel/sdxl_ball2.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">a red cat playing with a (ball)0.6</figcaption>
  </div>
</div>

We can see that the ball is almost completely gone on the right image while it's clearly visible on the left image.
For more information and more tricks you can use `compel` with, please have a look at the [compel docs](https://github.com/damian0815/compel/blob/main/doc/syntax.md) as well.
