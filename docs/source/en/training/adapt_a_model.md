# Adapt a model to a new task

Many diffusion systems share the same components, allowing you to adapt a pretrained model for one task to an entirely different task.

This guide will show you how to adapt a pretrained text-to-image model for inpainting by initializing and modifying the architecture of a pretrained [`UNet2DConditionModel`].

## Configure UNet2DConditionModel parameters

A [`UNet2DConditionModel`] by default accepts 4 channels in the [input sample](https://huggingface.co/docs/diffusers/v0.16.0/en/api/models#diffusers.UNet2DConditionModel.in_channels). For example, load a pretrained text-to-image model like [`stable-diffusion-v1-5/stable-diffusion-v1-5`](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) and take a look at the number of `in_channels`:

```py
from diffusers import StableDiffusionPipeline

pipeline = StableDiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", use_safetensors=True)
pipeline.unet.config["in_channels"]
4
```

Inpainting requires 9 channels in the input sample. You can check this value in a pretrained inpainting model like [`runwayml/stable-diffusion-inpainting`](https://huggingface.co/runwayml/stable-diffusion-inpainting):

```py
from diffusers import StableDiffusionPipeline

pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-inpainting", use_safetensors=True)
pipeline.unet.config["in_channels"]
9
```

To adapt your text-to-image model for inpainting, you'll need to change the number of `in_channels` from 4 to 9.

Initialize a [`UNet2DConditionModel`] with the pretrained text-to-image model weights, and change `in_channels` to 9. Changing the number of `in_channels` means you need to set `ignore_mismatched_sizes=True` and `low_cpu_mem_usage=False` to avoid a size mismatch error because the shape is different now.

```py
from diffusers import UNet2DConditionModel

model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
unet = UNet2DConditionModel.from_pretrained(
    model_id,
    subfolder="unet",
    in_channels=9,
    low_cpu_mem_usage=False,
    ignore_mismatched_sizes=True,
    use_safetensors=True,
)
```

The pretrained weights of the other components from the text-to-image model are initialized from their checkpoints, but the input channel weights (`conv_in.weight`) of the `unet` are randomly initialized. It is important to finetune the model for inpainting because otherwise the model returns noise.
