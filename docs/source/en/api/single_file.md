# Loading Pipelines and Models via `from_single_file`

The `from_single_file` method allows you to load supported Pipelines using a single checkpoint file as opposed to the folder format used by Diffusers. This is useful if you are working with many of the Stable Diffusion Web UI's that extensively rely on a single file to distribute all the components of a Diffusion Model.

The `from_single_file` method also supports loading models in their originally distributed format. This means that supported models that have been finetuned with other services can be loaded directly into supported Diffusers model objects and Pipelines.

## Pipelines that currently support `from_single_file` loading

- [StableDiffusionPipeline]()
- [StableDiffusionImg2ImgPipeline]()
- [StableDiffusionInpaintPipeline]()
- [StableDiffusionControlNetPipeline]()
- [StableDiffusionControlNetImg2ImgPipeline]()
- [StableDiffusionControlNetInpaintPipeline]()
- [StableDiffusionUpscalePipeline]()
- [StableDiffusionXLPipeline]()
- [StableDiffusionXLImg2ImgPipeline]()
- [StableDiffusionXLInpaintPipeline]()
- [StableDiffusionXLControlNetPipeline]()

## Models that currently support `from_single_file` loading

- [UNet2DConditionModel]()
- [StableCascadeUNet]()
- [AutoencoderKL]()
- [ControlNetModel]()

## Usage Examples

## Loading a Pipeline using `from_single_file`

```python
from diffusers import StableDiffusionXLPipeline

ckpt_path = "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/sd_xl_base_1.0_0.9vae.safetensors"
pipe = StableDiffusionXLPipeline.from_single_file(ckpt_path)
```

## Setting components in a Pipeline using `from_single_file`

```python
from diffusers import StableDiffusionXLPipeline, DDIMScheduler

ckpt_path = "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/sd_xl_base_1.0_0.9vae.safetensors"

scheduler = DDIMScheduler()
pipe = StableDiffusionXLPipeline.from_single_file(ckpt_path, scheduler=scheduler)
```

```python
from diffusers import StableDiffusionPipeline, ControlNetModel

ckpt_path = "https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/v1-5-pruned-emaonly.safetensors"

controlnet = ControlNetModel.from_pretrained("https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/v1-5-pruned-emaonly.safetensors")
pipe = StableDiffusionPipeline.from_single_file(ckpt_path, controlnet=controlnet)
```

## Loading a Model using `from_single_file`

```python
from diffusers import StableCascadeUNet

ckpt_path = "https://huggingface.co/stabilityai/stable-cascade/blob/main/stage_b_lite.safetensors"
model = StableCascadeUNet.from_single_file(ckpt_path)
```

## Override configuration options when using single file loading

In order to override configuration options when loading a model or pipeline using `from_single_file`, you can pass the relevant arguments directly to the `from_single_file` method.

```python
from diffusers import UNet2DConditionModel

ckpt_path = "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/sd_xl_base_1.0_0.9vae.safetensors"
model = UNet2DConditionModel.from_single_file(ckpt_path, device="cuda", upcast_attention=True)
```

## Downloading a single file checkpoint to a specific directory

```python
from diffusers import StableDiffusionXLPipeline

ckpt_path = "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/sd_xl_base_1.0_0.9vae.safetensors"
pipe = StableDiffusionXLPipeline.from_single_file(ckpt_path, local_dir="my_checkpoints")
```

## Disable symlinking when downloading a single file checkpoint

```python
from diffusers import StableDiffusionXLPipeline

ckpt_path = "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/sd_xl_base_1.0_0.9vae.safetensors"
pipe = StableDiffusionXLPipeline.from_single_file(ckpt_path, local_dir="my_checkpoints", local_dir_use_symlinks=False)
```
