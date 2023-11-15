<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Community pipelines

[[open-in-colab]]

<Tip>

For more context about the design choices behind community pipelines, please have a look at [this issue](https://github.com/huggingface/diffusers/issues/841).

</Tip>

Community pipelines allow you to get creative and build your own unique pipelines to share with the community. You can find all community pipelines in the [diffusers/examples/community](https://github.com/huggingface/diffusers/tree/main/examples/community) folder along with inference and training examples for how to use them. This guide showcases some of the community pipelines and hopefully it'll inspire you to create your own (feel free to open a PR with your own pipeline and we will merge it!).

To load a community pipeline, use the `custom_pipeline` argument in [`DiffusionPipeline`] to specify one of the files in [diffusers/examples/community](https://github.com/huggingface/diffusers/tree/main/examples/community):

```py
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4", custom_pipeline="filename_in_the_community_folder", use_safetensors=True
)
```

If a community pipeline doesn't work as expected, please open a GitHub issue and mention the author.

You can learn more about community pipelines in the how to [load community pipelines](custom_pipeline_overview) and how to [contribute a community pipeline](contribute_pipeline) guides.

## Multilingual Stable Diffusion

The multilingual Stable Diffusion pipeline uses a pretrained [XLM-RoBERTa](https://huggingface.co/papluca/xlm-roberta-base-language-detection) to identify a language and the [mBART-large-50](https://huggingface.co/facebook/mbart-large-50-many-to-one-mmt) model to handle the translation. This allows you to generate images from text in 20 languages.

```py
import torch
from diffusers import DiffusionPipeline
from diffusers.utils import make_image_grid
from transformers import (
    pipeline,
    MBart50TokenizerFast,
    MBartForConditionalGeneration,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
device_dict = {"cuda": 0, "cpu": -1}

# add language detection pipeline
language_detection_model_ckpt = "papluca/xlm-roberta-base-language-detection"
language_detection_pipeline = pipeline("text-classification",
                                       model=language_detection_model_ckpt,
                                       device=device_dict[device])

# add model for language translation
translation_tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-one-mmt")
translation_model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-one-mmt").to(device)

diffuser_pipeline = DiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    custom_pipeline="multilingual_stable_diffusion",
    detection_pipeline=language_detection_pipeline,
    translation_model=translation_model,
    translation_tokenizer=translation_tokenizer,
    torch_dtype=torch.float16,
)

diffuser_pipeline.enable_attention_slicing()
diffuser_pipeline = diffuser_pipeline.to(device)

prompt = ["a photograph of an astronaut riding a horse",
          "Una casa en la playa",
          "Ein Hund, der Orange isst",
          "Un restaurant parisien"]

images = diffuser_pipeline(prompt).images
make_image_grid(images, rows=2, cols=2)
```

<div class="flex justify-center">
    <img src="https://user-images.githubusercontent.com/4313860/198328706-295824a4-9856-4ce5-8e66-278ceb42fd29.png"/>
</div>

## MagicMix

[MagicMix](https://huggingface.co/papers/2210.16056) is a pipeline that can mix an image and text prompt to generate a new image that preserves the image structure. The `mix_factor` determines how much influence the prompt has on the layout generation, `kmin` controls the number of steps during the content generation process, and `kmax` determines how much information is kept in the layout of the original image.

```py
from diffusers import DiffusionPipeline, DDIMScheduler
from diffusers.utils import load_image, make_image_grid

pipeline = DiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    custom_pipeline="magic_mix",
    scheduler=DDIMScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler"),
).to('cuda')

img = load_image("https://user-images.githubusercontent.com/59410571/209578593-141467c7-d831-4792-8b9a-b17dc5e47816.jpg")
mix_img = pipeline(img, prompt="bed", kmin=0.3, kmax=0.5, mix_factor=0.5)
make_image_grid([img, mix_img], rows=1, cols=2)
```

<div class="flex gap-4">
  <div>
    <img class="rounded-xl" src="https://user-images.githubusercontent.com/59410571/209578593-141467c7-d831-4792-8b9a-b17dc5e47816.jpg" />
    <figcaption class="mt-2 text-center text-sm text-gray-500">original image</figcaption>
  </div>
  <div>
    <img class="rounded-xl" src="https://user-images.githubusercontent.com/59410571/209578602-70f323fa-05b7-4dd6-b055-e40683e37914.jpg" />
    <figcaption class="mt-2 text-center text-sm text-gray-500">image and text prompt mix</figcaption>
  </div>
</div>
