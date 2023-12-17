<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# UniDiffuser

The UniDiffuser model was proposed in [One Transformer Fits All Distributions in Multi-Modal Diffusion at Scale](https://huggingface.co/papers/2303.06555) by Fan Bao, Shen Nie, Kaiwen Xue, Chongxuan Li, Shi Pu, Yaole Wang, Gang Yue, Yue Cao, Hang Su, Jun Zhu.

The abstract from the paper is:

*This paper proposes a unified diffusion framework (dubbed UniDiffuser) to fit all distributions relevant to a set of multi-modal data in one model. Our key insight is -- learning diffusion models for marginal, conditional, and joint distributions can be unified as predicting the noise in the perturbed data, where the perturbation levels (i.e. timesteps) can be different for different modalities. Inspired by the unified view, UniDiffuser learns all distributions simultaneously with a minimal modification to the original diffusion model -- perturbs data in all modalities instead of a single modality, inputs individual timesteps in different modalities, and predicts the noise of all modalities instead of a single modality. UniDiffuser is parameterized by a transformer for diffusion models to handle input types of different modalities. Implemented on large-scale paired image-text data, UniDiffuser is able to perform image, text, text-to-image, image-to-text, and image-text pair generation by setting proper timesteps without additional overhead. In particular, UniDiffuser is able to produce perceptually realistic samples in all tasks and its quantitative results (e.g., the FID and CLIP score) are not only superior to existing general-purpose models but also comparable to the bespoken models (e.g., Stable Diffusion and DALL-E 2) in representative tasks (e.g., text-to-image generation).*

You can find the original codebase at [thu-ml/unidiffuser](https://github.com/thu-ml/unidiffuser) and additional checkpoints at [thu-ml](https://huggingface.co/thu-ml).

<Tip warning={true}>

There is currently an issue on PyTorch 1.X where the output images are all black or the pixel values become `NaNs`. This issue can be mitigated by switching to PyTorch 2.X.

</Tip>

This pipeline was contributed by [dg845](https://github.com/dg845). ❤️

## Usage Examples

Because the UniDiffuser model is trained to model the joint distribution of (image, text) pairs, it is capable of performing a diverse range of generation tasks:

### Unconditional Image and Text Generation

Unconditional generation (where we start from only latents sampled from a standard Gaussian prior) from a [`UniDiffuserPipeline`] will produce a (image, text) pair:

```python
import torch

from diffusers import UniDiffuserPipeline

device = "cuda"
model_id_or_path = "thu-ml/unidiffuser-v1"
pipe = UniDiffuserPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16)
pipe.to(device)

# Unconditional image and text generation. The generation task is automatically inferred.
sample = pipe(num_inference_steps=20, guidance_scale=8.0)
image = sample.images[0]
text = sample.text[0]
image.save("unidiffuser_joint_sample_image.png")
print(text)
```

This is also called "joint" generation in the UniDiffuser paper, since we are sampling from the joint image-text distribution.

Note that the generation task is inferred from the inputs used when calling the pipeline.
It is also possible to manually specify the unconditional generation task ("mode") manually with [`UniDiffuserPipeline.set_joint_mode`]:

```python
# Equivalent to the above.
pipe.set_joint_mode()
sample = pipe(num_inference_steps=20, guidance_scale=8.0)
```

When the mode is set manually, subsequent calls to the pipeline will use the set mode without attempting to infer the mode.
You can reset the mode with [`UniDiffuserPipeline.reset_mode`], after which the pipeline will once again infer the mode.

You can also generate only an image or only text (which the UniDiffuser paper calls "marginal" generation since we sample from the marginal distribution of images and text, respectively):

```python
# Unlike other generation tasks, image-only and text-only generation don't use classifier-free guidance
# Image-only generation
pipe.set_image_mode()
sample_image = pipe(num_inference_steps=20).images[0]
# Text-only generation
pipe.set_text_mode()
sample_text = pipe(num_inference_steps=20).text[0]
```

### Text-to-Image Generation

UniDiffuser is also capable of sampling from conditional distributions; that is, the distribution of images conditioned on a text prompt or the distribution of texts conditioned on an image.
Here is an example of sampling from the conditional image distribution (text-to-image generation or text-conditioned image generation):

```python
import torch

from diffusers import UniDiffuserPipeline

device = "cuda"
model_id_or_path = "thu-ml/unidiffuser-v1"
pipe = UniDiffuserPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16)
pipe.to(device)

# Text-to-image generation
prompt = "an elephant under the sea"

sample = pipe(prompt=prompt, num_inference_steps=20, guidance_scale=8.0)
t2i_image = sample.images[0]
t2i_image
```

The `text2img` mode requires that either an input `prompt` or `prompt_embeds` be supplied. You can set the `text2img` mode manually with [`UniDiffuserPipeline.set_text_to_image_mode`].

### Image-to-Text Generation

Similarly, UniDiffuser can also produce text samples given an image (image-to-text or image-conditioned text generation):

```python
import torch

from diffusers import UniDiffuserPipeline
from diffusers.utils import load_image

device = "cuda"
model_id_or_path = "thu-ml/unidiffuser-v1"
pipe = UniDiffuserPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16)
pipe.to(device)

# Image-to-text generation
image_url = "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/unidiffuser/unidiffuser_example_image.jpg"
init_image = load_image(image_url).resize((512, 512))

sample = pipe(image=init_image, num_inference_steps=20, guidance_scale=8.0)
i2t_text = sample.text[0]
print(i2t_text)
```

The `img2text` mode requires that an input `image` be supplied. You can set the `img2text` mode manually with [`UniDiffuserPipeline.set_image_to_text_mode`].

### Image Variation

The UniDiffuser authors suggest performing image variation through a "round-trip" generation method, where given an input image, we first perform an image-to-text generation, and then perform a text-to-image generation on the outputs of the first generation.
This produces a new image which is semantically similar to the input image:

```python
import torch

from diffusers import UniDiffuserPipeline
from diffusers.utils import load_image

device = "cuda"
model_id_or_path = "thu-ml/unidiffuser-v1"
pipe = UniDiffuserPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16)
pipe.to(device)

# Image variation can be performed with an image-to-text generation followed by a text-to-image generation:
# 1. Image-to-text generation
image_url = "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/unidiffuser/unidiffuser_example_image.jpg"
init_image = load_image(image_url).resize((512, 512))

sample = pipe(image=init_image, num_inference_steps=20, guidance_scale=8.0)
i2t_text = sample.text[0]
print(i2t_text)

# 2. Text-to-image generation
sample = pipe(prompt=i2t_text, num_inference_steps=20, guidance_scale=8.0)
final_image = sample.images[0]
final_image.save("unidiffuser_image_variation_sample.png")
```

### Text Variation

Similarly, text variation can be performed on an input prompt with a text-to-image generation followed by a image-to-text generation:

```python
import torch

from diffusers import UniDiffuserPipeline

device = "cuda"
model_id_or_path = "thu-ml/unidiffuser-v1"
pipe = UniDiffuserPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16)
pipe.to(device)

# Text variation can be performed with a text-to-image generation followed by a image-to-text generation:
# 1. Text-to-image generation
prompt = "an elephant under the sea"

sample = pipe(prompt=prompt, num_inference_steps=20, guidance_scale=8.0)
t2i_image = sample.images[0]
t2i_image.save("unidiffuser_text2img_sample_image.png")

# 2. Image-to-text generation
sample = pipe(image=t2i_image, num_inference_steps=20, guidance_scale=8.0)
final_prompt = sample.text[0]
print(final_prompt)
```

<Tip>

Make sure to check out the Schedulers [guide](../../using-diffusers/schedulers) to learn how to explore the tradeoff between scheduler speed and quality, and see the [reuse components across pipelines](../../using-diffusers/loading#reuse-components-across-pipelines) section to learn how to efficiently load the same components into multiple pipelines.

</Tip>

## UniDiffuserPipeline
[[autodoc]] UniDiffuserPipeline
	- all
	- __call__

## ImageTextPipelineOutput
[[autodoc]] pipelines.ImageTextPipelineOutput
