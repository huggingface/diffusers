# Community Examples

> **For more information about community pipelines, please have a look at [this issue](https://github.com/huggingface/diffusers/issues/841).**

**Community** examples consist of both inference and training examples that have been added by the community.

| Example   |      Description      |      Author      |    Colab  |
|:----------|:----------------------|:-----------------|----------:|
| CLIP Guided Stable Diffusion | Doing CLIP guidance for text to image generation with Stable Diffusion| [Suraj Patil](https://github.com/patil-suraj/) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/CLIP_Guided_Stable_diffusion_with_diffusers.ipynb) |
| One Step U-Net (Dummy) | Example showcasing of how to use Community Pipelines (see https://github.com/huggingface/diffusers/issues/841) | [Patrick von Platen](https://github.com/patrickvonplaten/) | - |
| Stable Diffusion Interpolation | Interpolate the latent space of Stable Diffusion between different prompts/seeds | [Nate Raw](https://github.com/nateraw/) | - |

## Example usages

### CLIP Guided Stable Diffusion

CLIP guided stable diffusion can help to generate more realistic images 
by guiding stable diffusion at every denoising step with an additional CLIP model.

The following code requires roughly 12GB of GPU RAM.

```python
from diffusers import DiffusionPipeline
from transformers import CLIPFeatureExtractor, CLIPModel
import torch


feature_extractor = CLIPFeatureExtractor.from_pretrained("laion/CLIP-ViT-B-32-laion2B-s34B-b79K")
clip_model = CLIPModel.from_pretrained("laion/CLIP-ViT-B-32-laion2B-s34B-b79K", torch_dtype=torch.float16)


guided_pipeline = DiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    custom_pipeline="clip_guided_stable_diffusion",
    clip_model=clip_model,
    feature_extractor=feature_extractor,
    revision="fp16",
    torch_dtype=torch.float16,
)
guided_pipeline.enable_attention_slicing()
guided_pipeline = guided_pipeline.to("cuda")

prompt = "fantasy book cover, full moon, fantasy forest landscape, golden vector elements, fantasy magic, dark light night, intricate, elegant, sharp focus, illustration, highly detailed, digital painting, concept art, matte, art by WLOP and Artgerm and Albert Bierstadt, masterpiece"

generator = torch.Generator(device="cuda").manual_seed(0)
images = []
for i in range(4):
    image = guided_pipeline(
        prompt,
        num_inference_steps=50,
        guidance_scale=7.5,
        clip_guidance_scale=100,
        num_cutouts=4,
        use_cutouts=False,
        generator=generator,
    ).images[0]
    images.append(image)
    
# save images locally
for i, img in enumerate(images):
    img.save(f"./clip_guided_sd/image_{i}.png")
```

The `images` list contains a list of PIL images that can be saved locally or displayed directly in a google colab.
Generated images tend to be of higher qualtiy than natively using stable diffusion. E.g. the above script generates the following images:

![clip_guidance](https://huggingface.co/datasets/patrickvonplaten/images/resolve/main/clip_guidance/merged_clip_guidance.jpg).

### One Step U-Net (Dummy)

The dummy "one-step-unet" can be run as follows:

```python
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained("google/ddpm-cifar10-32", custom_pipeline="one_step_unet")
pipe()
```

**Note**: This community pipeline is not useful as a feature, but rather just serves as an example of how community pipelines can be added (see https://github.com/huggingface/diffusers/issues/841).

### Stable Diffusion Interpolation

The following code can be run on a GPU of at least 8GB VRAM and should take approximately 5 minutes.

```python
from diffusers import DiffusionPipeline
import torch

pipe = DiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    revision='fp16',
    torch_dtype=torch.float16,
    safety_checker=None,  # Very important for videos...lots of false positives while interpolating
    custom_pipeline="interpolate_stable_diffusion",
).to('cuda')
pipe.enable_attention_slicing()

frame_filepaths = pipe.walk(
    prompts=['a dog', 'a cat', 'a horse'],
    seeds=[42, 1337, 1234],
    num_interpolation_steps=16,
    output_dir='./dreams',
    batch_size=4,
    height=512,
    width=512,
    guidance_scale=8.5,
    num_inference_steps=50,
)
```

The output of the `walk(...)` function returns a list of images saved under the folder as defined in `output_dir`. You can use these images to create videos of stable diffusion. 

> **Please have a look at https://github.com/nateraw/stable-diffusion-videos for more in-detail information on how to create videos using stable diffusion as well as more feature-complete functionality.**
