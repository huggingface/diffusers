<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Load community pipelines and components

[[open-in-colab]]

## Community pipelines

> [!TIP] Take a look at GitHub Issue [#841](https://github.com/huggingface/diffusers/issues/841) for more context about why we're adding community pipelines to help everyone easily share their work without being slowed down.

Community pipelines are any [`DiffusionPipeline`] class that are different from the original paper implementation (for example, the [`StableDiffusionControlNetPipeline`] corresponds to the [Text-to-Image Generation with ControlNet Conditioning](https://arxiv.org/abs/2302.05543) paper). They provide additional functionality or extend the original implementation of a pipeline.

There are many cool community pipelines like [Marigold Depth Estimation](https://github.com/huggingface/diffusers/tree/main/examples/community#marigold-depth-estimation) or [InstantID](https://github.com/huggingface/diffusers/tree/main/examples/community#instantid-pipeline), and you can find all the official community pipelines [here](https://github.com/huggingface/diffusers/tree/main/examples/community).

There are two types of community pipelines, those stored on the Hugging Face Hub and those stored on Diffusers GitHub repository. Hub pipelines are completely customizable (scheduler, models, pipeline code, etc.) while Diffusers GitHub pipelines are only limited to custom pipeline code.

|                | GitHub community pipeline                                                                                        | HF Hub community pipeline                                                                 |
|----------------|------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------|
| usage          | same                                                                                                             | same                                                                                      |
| review process | open a Pull Request on GitHub and undergo a review process from the Diffusers team before merging; may be slower | upload directly to a Hub repository without any review; this is the fastest workflow      |
| visibility     | included in the official Diffusers repository and documentation                                                  | included on your HF Hub profile and relies on your own usage/promotion to gain visibility |

<hfoptions id="community">
<hfoption id="Hub pipelines">

To load a Hugging Face Hub community pipeline, pass the repository id of the community pipeline to the `custom_pipeline` argument and the model repository where you'd like to load the pipeline weights and components from. For example, the example below loads a dummy pipeline from [hf-internal-testing/diffusers-dummy-pipeline](https://huggingface.co/hf-internal-testing/diffusers-dummy-pipeline/blob/main/pipeline.py) and the pipeline weights and components from [google/ddpm-cifar10-32](https://huggingface.co/google/ddpm-cifar10-32):

> [!WARNING]
> By loading a community pipeline from the Hugging Face Hub, you are trusting that the code you are loading is safe. Make sure to inspect the code online before loading and running it automatically!

```py
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
    "google/ddpm-cifar10-32", custom_pipeline="hf-internal-testing/diffusers-dummy-pipeline", use_safetensors=True
)
```

</hfoption>
<hfoption id="GitHub pipelines">

To load a GitHub community pipeline, pass the repository id of the community pipeline to the `custom_pipeline` argument and the model repository where you you'd like to load the pipeline weights and components from. You can also load model components directly. The example below loads the community [CLIP Guided Stable Diffusion](https://github.com/huggingface/diffusers/tree/main/examples/community#clip-guided-stable-diffusion) pipeline and the CLIP model components.

```py
from diffusers import DiffusionPipeline
from transformers import CLIPImageProcessor, CLIPModel

clip_model_id = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"

feature_extractor = CLIPImageProcessor.from_pretrained(clip_model_id)
clip_model = CLIPModel.from_pretrained(clip_model_id)

pipeline = DiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    custom_pipeline="clip_guided_stable_diffusion",
    clip_model=clip_model,
    feature_extractor=feature_extractor,
    use_safetensors=True,
)
```

</hfoption>
</hfoptions>

### Load from a local file

Community pipelines can also be loaded from a local file if you pass a file path instead. The path to the passed directory must contain a pipeline.py file that contains the pipeline class.

```py
pipeline = DiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    custom_pipeline="./path/to/pipeline_directory/",
    clip_model=clip_model,
    feature_extractor=feature_extractor,
    use_safetensors=True,
)
```

### Load from a specific version

By default, community pipelines are loaded from the latest stable version of Diffusers. To load a community pipeline from another version, use the `custom_revision` parameter.

<hfoptions id="version">
<hfoption id="main">

For example, to load from the main branch:

```py
pipeline = DiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    custom_pipeline="clip_guided_stable_diffusion",
    custom_revision="main",
    clip_model=clip_model,
    feature_extractor=feature_extractor,
    use_safetensors=True,
)
```

</hfoption>
<hfoption id="older version">

For example, to load from a previous version of Diffusers like v0.25.0:

```py
pipeline = DiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    custom_pipeline="clip_guided_stable_diffusion",
    custom_revision="v0.25.0",
    clip_model=clip_model,
    feature_extractor=feature_extractor,
    use_safetensors=True,
)
```

</hfoption>
</hfoptions>

### Load with from_pipe

Community pipelines can also be loaded with the [`~DiffusionPipeline.from_pipe`] method which allows you to load and reuse multiple pipelines without any additional memory overhead (learn more in the [Reuse a pipeline](./loading#reuse-a-pipeline) guide). The memory requirement is determined by the largest single pipeline loaded.

For example, let's load a community pipeline that supports [long prompts with weighting](https://github.com/huggingface/diffusers/tree/main/examples/community#long-prompt-weighting-stable-diffusion) from a Stable Diffusion pipeline.

```py
import torch
from diffusers import DiffusionPipeline

pipe_sd = DiffusionPipeline.from_pretrained("emilianJR/CyberRealistic_V3", torch_dtype=torch.float16)
pipe_sd.to("cuda")
# load long prompt weighting pipeline
pipe_lpw = DiffusionPipeline.from_pipe(
    pipe_sd,
    custom_pipeline="lpw_stable_diffusion",
).to("cuda")

prompt = "cat, hiding in the leaves, ((rain)), zazie rainyday, beautiful eyes, macro shot, colorful details, natural lighting, amazing composition, subsurface scattering, amazing textures, filmic, soft light, ultra-detailed eyes, intricate details, detailed texture, light source contrast, dramatic shadows, cinematic light, depth of field, film grain, noise, dark background, hyperrealistic dslr film still, dim volumetric cinematic lighting"
neg_prompt = "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, mutated hands and fingers:1.4), (deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, disconnected limbs, mutation, mutated, ugly, disgusting, amputation"
generator = torch.Generator(device="cpu").manual_seed(20)
out_lpw = pipe_lpw(
    prompt,
    negative_prompt=neg_prompt,
    width=512,
    height=512,
    max_embeddings_multiples=3,
    num_inference_steps=50,
    generator=generator,
    ).images[0]
out_lpw
```

<div class="flex gap-4">
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/from_pipe_lpw.png" />
    <figcaption class="mt-2 text-center text-sm text-gray-500">Stable Diffusion with long prompt weighting</figcaption>
  </div>
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/from_pipe_non_lpw.png" />
    <figcaption class="mt-2 text-center text-sm text-gray-500">Stable Diffusion</figcaption>
  </div>
</div>

## Example community pipelines

Community pipelines are a really fun and creative way to extend the capabilities of the original pipeline with new and unique features. You can find all community pipelines in the [diffusers/examples/community](https://github.com/huggingface/diffusers/tree/main/examples/community) folder with inference and training examples for how to use them.

This section showcases a couple of the community pipelines and hopefully it'll inspire you to create your own (feel free to open a PR for your community pipeline and ping us for a review)!

> [!TIP]
> The [`~DiffusionPipeline.from_pipe`] method is particularly useful for loading community pipelines because many of them don't have pretrained weights and add a feature on top of an existing pipeline like Stable Diffusion or Stable Diffusion XL. You can learn more about the [`~DiffusionPipeline.from_pipe`] method in the [Load with from_pipe](custom_pipeline_overview#load-with-from_pipe) section.

<hfoptions id="community">
<hfoption id="Marigold">

[Marigold](https://marigoldmonodepth.github.io/) is a depth estimation diffusion pipeline that uses the rich existing and inherent visual knowledge in diffusion models. It takes an input image and denoises and decodes it into a depth map. Marigold performs well even on images it hasn't seen before.

```py
import torch
from PIL import Image
from diffusers import DiffusionPipeline
from diffusers.utils import load_image

pipeline = DiffusionPipeline.from_pretrained(
    "prs-eth/marigold-lcm-v1-0",
    custom_pipeline="marigold_depth_estimation",
    torch_dtype=torch.float16,
    variant="fp16",
)

pipeline.to("cuda")
image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/community-marigold.png")
output = pipeline(
    image,
    denoising_steps=4,
    ensemble_size=5,
    processing_res=768,
    match_input_res=True,
    batch_size=0,
    seed=33,
    color_map="Spectral",
    show_progress_bar=True,
)
depth_colored: Image.Image = output.depth_colored
depth_colored.save("./depth_colored.png")
```

<div class="flex flex-row gap-4">
  <div class="flex-1">
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/community-marigold.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">original image</figcaption>
  </div>
  <div class="flex-1">
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/marigold-depth.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">colorized depth image</figcaption>
  </div>
</div>

</hfoption>
<hfoption id="HD-Painter">

[HD-Painter](https://hf.co/papers/2312.14091) is a high-resolution inpainting pipeline. It introduces a *Prompt-Aware Introverted Attention (PAIntA)* layer to better align a prompt with the area to be inpainted, and *Reweighting Attention Score Guidance (RASG)* to keep the latents more prompt-aligned and within their trained domain to generate realistc images.

```py
import torch
from diffusers import DiffusionPipeline, DDIMScheduler
from diffusers.utils import load_image

pipeline = DiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5-inpainting",
    custom_pipeline="hd_painter"
)
pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
init_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/hd-painter.jpg")
mask_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/hd-painter-mask.png")
prompt = "football"
image = pipeline(prompt, init_image, mask_image, use_rasg=True, use_painta=True, generator=torch.manual_seed(0)).images[0]
image
```

<div class="flex flex-row gap-4">
  <div class="flex-1">
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/hd-painter.jpg"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">original image</figcaption>
  </div>
  <div class="flex-1">
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/hd-painter-output.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">generated image</figcaption>
  </div>
</div>

</hfoption>
</hfoptions>

## Community components

Community components allow users to build pipelines that may have customized components that are not a part of Diffusers. If your pipeline has custom components that Diffusers doesn't already support, you need to provide their implementations as Python modules. These customized components could be a VAE, UNet, and scheduler. In most cases, the text encoder is imported from the Transformers library. The pipeline code itself can also be customized.

This section shows how users should use community components to build a community pipeline.

You'll use the [showlab/show-1-base](https://huggingface.co/showlab/show-1-base) pipeline checkpoint as an example.

1. Import and load the text encoder from Transformers:

```python
from transformers import T5Tokenizer, T5EncoderModel

pipe_id = "showlab/show-1-base"
tokenizer = T5Tokenizer.from_pretrained(pipe_id, subfolder="tokenizer")
text_encoder = T5EncoderModel.from_pretrained(pipe_id, subfolder="text_encoder")
```

2. Load a scheduler:

```python
from diffusers import DPMSolverMultistepScheduler

scheduler = DPMSolverMultistepScheduler.from_pretrained(pipe_id, subfolder="scheduler")
```

3. Load an image processor:

```python
from transformers import CLIPImageProcessor

feature_extractor = CLIPImageProcessor.from_pretrained(pipe_id, subfolder="feature_extractor")
```

<Tip warning={true}>

In steps 4 and 5, the custom [UNet](https://github.com/showlab/Show-1/blob/main/showone/models/unet_3d_condition.py) and [pipeline](https://huggingface.co/sayakpaul/show-1-base-with-code/blob/main/unet/showone_unet_3d_condition.py) implementation must match the format shown in their files for this example to work.

</Tip>

4. Now you'll load a [custom UNet](https://github.com/showlab/Show-1/blob/main/showone/models/unet_3d_condition.py), which in this example, has already been implemented in [showone_unet_3d_condition.py](https://huggingface.co/sayakpaul/show-1-base-with-code/blob/main/unet/showone_unet_3d_condition.py) for your convenience. You'll notice the [`UNet3DConditionModel`] class name is changed to `ShowOneUNet3DConditionModel` because [`UNet3DConditionModel`] already exists in Diffusers. Any components needed for the `ShowOneUNet3DConditionModel` class should be placed in showone_unet_3d_condition.py.

    Once this is done, you can initialize the UNet:

    ```python
    from showone_unet_3d_condition import ShowOneUNet3DConditionModel

    unet = ShowOneUNet3DConditionModel.from_pretrained(pipe_id, subfolder="unet")
    ```

5. Finally, you'll load the custom pipeline code. For this example, it has already been created for you in [pipeline_t2v_base_pixel.py](https://huggingface.co/sayakpaul/show-1-base-with-code/blob/main/pipeline_t2v_base_pixel.py). This script contains a custom `TextToVideoIFPipeline` class for generating videos from text. Just like the custom UNet, any code needed for the custom pipeline to work should go in pipeline_t2v_base_pixel.py.

Once everything is in place, you can initialize the `TextToVideoIFPipeline` with the `ShowOneUNet3DConditionModel`:

```python
from pipeline_t2v_base_pixel import TextToVideoIFPipeline
import torch

pipeline = TextToVideoIFPipeline(
    unet=unet,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    scheduler=scheduler,
    feature_extractor=feature_extractor
)
pipeline = pipeline.to(device="cuda")
pipeline.torch_dtype = torch.float16
```

Push the pipeline to the Hub to share with the community!

```python
pipeline.push_to_hub("custom-t2v-pipeline")
```

After the pipeline is successfully pushed, you need to make a few changes:

1. Change the `_class_name` attribute in [model_index.json](https://huggingface.co/sayakpaul/show-1-base-with-code/blob/main/model_index.json#L2) to `"pipeline_t2v_base_pixel"` and `"TextToVideoIFPipeline"`.
2. Upload `showone_unet_3d_condition.py` to the [unet](https://huggingface.co/sayakpaul/show-1-base-with-code/blob/main/unet/showone_unet_3d_condition.py) subfolder.
3. Upload `pipeline_t2v_base_pixel.py` to the pipeline [repository](https://huggingface.co/sayakpaul/show-1-base-with-code/tree/main).

To run inference, add the `trust_remote_code` argument while initializing the pipeline to handle all the "magic" behind the scenes.

> [!WARNING]
> As an additional precaution with `trust_remote_code=True`, we strongly encourage you to pass a commit hash to the `revision` parameter in [`~DiffusionPipeline.from_pretrained`] to make sure the code hasn't been updated with some malicious new lines of code (unless you fully trust the model owners).

```python
from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained(
    "<change-username>/<change-id>", trust_remote_code=True, torch_dtype=torch.float16
).to("cuda")

prompt = "hello"

# Text embeds
prompt_embeds, negative_embeds = pipeline.encode_prompt(prompt)

# Keyframes generation (8x64x40, 2fps)
video_frames = pipeline(
    prompt_embeds=prompt_embeds,
    negative_prompt_embeds=negative_embeds,
    num_frames=8,
    height=40,
    width=64,
    num_inference_steps=2,
    guidance_scale=9.0,
    output_type="pt"
).frames
```

As an additional reference, take a look at the repository structure of [stabilityai/japanese-stable-diffusion-xl](https://huggingface.co/stabilityai/japanese-stable-diffusion-xl/) which also uses the `trust_remote_code` feature.

```python
from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained(
    "stabilityai/japanese-stable-diffusion-xl", trust_remote_code=True
)
pipeline.to("cuda")
```
