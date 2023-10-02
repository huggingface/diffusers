# FreeU to improve generation quality

[[open-in-colab]]

Usually a UNet is responsible for denoising during the reverse diffusion process. The features inside the UNet can be boradly classified distinctively: 

1. Backbone features
2. Skip features

In [FreeU: Free Lunch in Diffusion U-Net](https://hf.co/papers/2309.11497), Si et al. investigate the contributions of these features in the context of diffusion. They found out that backbone features primarily contribute to the denoising process while the skip features mainly introduce high-frequency features into the decoder module. Furthermore, the skip features can make the network overlook the semantics baked in the backbone features. 

To mitigate these issues, the authors introduce the **FreeU mechanism** where they simply reweigh the contributions sourced from the UNet’s skip connections and backbone feature maps, to leverage the strengths of both components. 

FreeU is an inference-time mechanism meaning that it does not require any additional training. It is completely technique that works with different tasks such as text-to-image, image-to-image, and text-to-video.

In this guide, we will discuss how to apply FreeU for different pipelines like [`StableDiffusionPipeline`], [`StableDiffusionXLPipeline`], and [`TextToVideoSDPipeline`].

## StableDiffusionPipeline

Load the pipeline: 

```py
from diffusers import DiffusionPipeline
import torch 

pipeline = DiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, safety_checker=None
).to("cuda")
```

Then enable the FreeU mechanism with the FreeU-specific hyperparameters:

```py
pipeline.enable_freeu(s1=0.9, s2=0.2, b1=1.2, b2=1.4)
```

Values for `s1`, `s2`, `b1`, and `b2` come from the official FreeU [code repository](https://github.com/ChenyangSi/FreeU). The authors also provide some guidance on the ranges of these hyperparameters [here](https://github.com/ChenyangSi/FreeU#range-for-more-parameters).  For more details on these hyperparameters, refer to the [original paper](https://hf.co/papers/2309.11497).

<Tip>

You can disable the FreeU mechanism by calling the `disable_freeu()` on a pipeline.

</Tip>

And then run inference:

```py
prompt = "A squirrel eating a burger"
seed = 2023
image = pipeline(prompt, generator=torch.manual_seed(seed)).images[0]
```

The figure below compares non-FreeU and FreeU results respectively for the same hyperparameters used above (`prompt` and `seed`):

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/freeu/sdv1_5_freeu.jpg)

We can clearly see that the results have improved with FreeU (especially for the final image).

Let's see how Stable Diffusion 2 results are impacted:

```py
from diffusers import DiffusionPipeline
import torch 

pipeline = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16, safety_checker=None
).to("cuda")

prompt = "A squirrel eating a burger"
seed = 2023

pipeline.enable_freeu(s1=0.9, s2=0.2, b1=1.1, b2=1.2)
image = pipeline(prompt, generator=torch.manual_seed(seed)).images[0]
```

Here is the non-FreeU vs. FreeU comparison:

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/freeu/sdv2_1_freeu.jpg)

## Stable Diffusion XL

```py
from diffusers import DiffusionPipeline
import torch 

pipeline = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16,
).to("cuda")

prompt = "A squirrel eating a burger"
seed = 2023

# Comes from 
# https://wandb.ai/nasirk24/UNET-FreeU-SDXL/reports/FreeU-SDXL-Optimal-Parameters--Vmlldzo1NDg4NTUw
pipeline.enable_freeu(s1=0.6, s2=0.4, b1=1.1, b2=1.2)
image = pipeline(prompt, generator=torch.manual_seed(seed)).images[0]
```

Here is the non-FreeU vs. FreeU comparison:

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/freeu/sdxl_freeu.jpg)

## Text-to-video generation

```python
from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video
import torch

model_id = "cerspense/zeroscope_v2_576w"
pipe = DiffusionPipeline.from_pretrained("cerspense/zeroscope_v2_576w", torch_dtype=torch.float16).to("cuda")
pipe = pipe.to("cuda")

prompt = "an astronaut riding a horse on mars"
seed = 2023

# The values come from
# https://github.com/lyn-rgb/FreeU_Diffusers#video-pipelines
pipe.enable_freeu(b1=1.2, b2=1.4, s1=0.9, s2=0.2)
video_frames = pipe(prompt, height=320, width=576, num_frames=30, generator=torch.manual_seed(seed)).frames
export_to_video(video_frames, "astronaut_rides_horse.mp4")
```

*Thanks [kadirnar](https://github.com/kadirnar/) for helping to integrate the feature. Thanks to [justindujardin](https://github.com/justindujardin) for the helpful discussions.*