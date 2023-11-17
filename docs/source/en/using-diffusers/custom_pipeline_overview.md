<!--Copyright 2023 The HuggingFace Team. All rights reserved.

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

Community pipelines are any [`DiffusionPipeline`] class that are different from the original implementation as specified in their paper (for example, the [`StableDiffusionControlNetPipeline`] corresponds to the [Text-to-Image Generation with ControlNet Conditioning](https://arxiv.org/abs/2302.05543) paper). They provide additional functionality or extend the original implementation of a pipeline.

There are many cool community pipelines like [Speech to Image](https://github.com/huggingface/diffusers/tree/main/examples/community#speech-to-image) or [Composable Stable Diffusion](https://github.com/huggingface/diffusers/tree/main/examples/community#composable-stable-diffusion), and you can find all the official community pipelines [here](https://github.com/huggingface/diffusers/tree/main/examples/community).

To load any community pipeline on the Hub, pass the repository id of the community pipeline to the `custom_pipeline` argument and the model repository where you'd like to load the pipeline weights and components from. For example, the example below loads a dummy pipeline from [`hf-internal-testing/diffusers-dummy-pipeline`](https://huggingface.co/hf-internal-testing/diffusers-dummy-pipeline/blob/main/pipeline.py) and the pipeline weights and components from [`google/ddpm-cifar10-32`](https://huggingface.co/google/ddpm-cifar10-32):

<Tip warning={true}>

🔒 By loading a community pipeline from the Hugging Face Hub, you are trusting that the code you are loading is safe. Make sure to inspect the code online before loading and running it automatically!

</Tip>

```py
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
    "google/ddpm-cifar10-32", custom_pipeline="hf-internal-testing/diffusers-dummy-pipeline", use_safetensors=True
)
```

Loading an official community pipeline is similar, but you can mix loading weights from an official repository id and pass pipeline components directly. The example below loads the community [CLIP Guided Stable Diffusion](https://github.com/huggingface/diffusers/tree/main/examples/community#clip-guided-stable-diffusion) pipeline, and you can pass the CLIP model components directly to it:

```py
from diffusers import DiffusionPipeline
from transformers import CLIPImageProcessor, CLIPModel

clip_model_id = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"

feature_extractor = CLIPImageProcessor.from_pretrained(clip_model_id)
clip_model = CLIPModel.from_pretrained(clip_model_id)

pipeline = DiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    custom_pipeline="clip_guided_stable_diffusion",
    clip_model=clip_model,
    feature_extractor=feature_extractor,
    use_safetensors=True,
)
```

For more information about community pipelines, take a look at the [Community pipelines](custom_pipeline_examples) guide for how to use them and if you're interested in adding a community pipeline check out the [How to contribute a community pipeline](contribute_pipeline) guide!

## Community components

Community components allow users to build pipelines that may have customized components that are not a part of Diffusers. If your pipeline has custom components that Diffusers doesn't already support, you need to provide their implementations as Python modules. These customized components could be a VAE, UNet, and scheduler. In most cases, the text encoder is imported from the Transformers library. The pipeline code itself can also be customized.

This section shows how users should use community components to build a community pipeline.

You'll use the [showlab/show-1-base](https://huggingface.co/showlab/show-1-base) pipeline checkpoint as an example. So, let's start loading the components:

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
from transformers import CLIPFeatureExtractor

feature_extractor = CLIPFeatureExtractor.from_pretrained(pipe_id, subfolder="feature_extractor")
```

<Tip warning={true}>

In steps 4 and 5, the custom [UNet](https://github.com/showlab/Show-1/blob/main/showone/models/unet_3d_condition.py) and [pipeline](https://huggingface.co/sayakpaul/show-1-base-with-code/blob/main/unet/showone_unet_3d_condition.py) implementation must match the format shown in their files for this example to work.

</Tip>

4. Now you'll load a [custom UNet](https://github.com/showlab/Show-1/blob/main/showone/models/unet_3d_condition.py), which in this example, has already been implemented in the `showone_unet_3d_condition.py` [script](https://huggingface.co/sayakpaul/show-1-base-with-code/blob/main/unet/showone_unet_3d_condition.py) for your convenience. You'll notice the `UNet3DConditionModel` class name is changed to `ShowOneUNet3DConditionModel` because [`UNet3DConditionModel`] already exists in Diffusers. Any components needed for the `ShowOneUNet3DConditionModel` class should be placed in the `showone_unet_3d_condition.py` script.

Once this is done, you can initialize the UNet:

```python
from showone_unet_3d_condition import ShowOneUNet3DConditionModel

unet = ShowOneUNet3DConditionModel.from_pretrained(pipe_id, subfolder="unet")
```

5. Finally, you'll load the custom pipeline code. For this example, it has already been created for you in the `pipeline_t2v_base_pixel.py` [script](https://huggingface.co/sayakpaul/show-1-base-with-code/blob/main/pipeline_t2v_base_pixel.py). This script contains a custom `TextToVideoIFPipeline` class for generating videos from text. Just like the custom UNet, any code needed for the custom pipeline to work should go in the `pipeline_t2v_base_pixel.py` script. 

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

After the pipeline is successfully pushed, you need a couple of changes:

1. Change the `_class_name` attribute in [`model_index.json`](https://huggingface.co/sayakpaul/show-1-base-with-code/blob/main/model_index.json#L2) to `"pipeline_t2v_base_pixel"` and `"TextToVideoIFPipeline"`.
2. Upload `showone_unet_3d_condition.py` to the `unet` [directory](https://huggingface.co/sayakpaul/show-1-base-with-code/blob/main/unet/showone_unet_3d_condition.py).
3. Upload `pipeline_t2v_base_pixel.py` to the pipeline base [directory](https://huggingface.co/sayakpaul/show-1-base-with-code/blob/main/unet/showone_unet_3d_condition.py).

To run inference, simply add the `trust_remote_code` argument while initializing the pipeline to handle all the "magic" behind the scenes.

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

As an additional reference example, you can refer to the repository structure of [stabilityai/japanese-stable-diffusion-xl](https://huggingface.co/stabilityai/japanese-stable-diffusion-xl/), that makes use of the `trust_remote_code` feature:

```python

from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained(
    "stabilityai/japanese-stable-diffusion-xl", trust_remote_code=True
)
pipeline.to("cuda")

# if using torch < 2.0
# pipeline.enable_xformers_memory_efficient_attention()

prompt = "柴犬、カラフルアート"

image = pipeline(prompt=prompt).images[0]

```