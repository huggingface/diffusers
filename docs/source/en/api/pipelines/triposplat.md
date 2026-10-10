# TripoSplat

[TripoSplat](https://github.com/VAST-AI-Research/TripoSplat) generates 3D Gaussian splats from foreground images.
Both the standard and modular APIs use the same converted components. DINOv3 and the image VAE use bfloat16 in
the reference workflow; the transformer, Gaussian decoder, and optional BiRefNet model use float16.

The pipeline uses `FlowMatchEulerDiscreteScheduler` with `shift=3.0` and supplies the reference sigma grid.
Predictions are cast to float32 for scheduler updates so Gaussian and camera latents remain in float32 throughout
denoising. Sigma storage and velocity updates use float32, which can produce numerical differences from the reference.

## Convert the checkpoint

The [official checkpoint](https://huggingface.co/VAST-AI/TripoSplat) must first be converted to Diffusers format.
Run the conversion script from a Diffusers checkout:

```bash
python scripts/convert_triposplat_to_diffusers.py \
    --output_dir ./triposplat-diffusers \
    --include_background_remover
```

The default `--dtype auto` preserves the reference component dtypes. An explicit `--dtype float16`,
`--dtype bfloat16`, or `--dtype float32` applies that dtype to every component.

Omit `--include_background_remover` to use images with a foreground alpha mask, or prepared RGB images with
`is_preprocessed=True`. BiRefNet is an optional component implemented under the TripoSplat pipeline package and
requires torchvision for deformable convolutions. It loads through the normal Diffusers model loading API.
Its Swin attention uses additive relative-position bias, so select a backend that supports additive masks, such
as native PyTorch SDPA.

## Standard pipeline

```python
import torch
from PIL import Image
from diffusers import TripoSplatPipeline
from diffusers.utils import export_to_gaussian_ply, export_to_splat

dtype = {"default": torch.float16, "image_encoder": torch.bfloat16, "vae": torch.bfloat16}
pipe = TripoSplatPipeline.from_pretrained("./triposplat-diffusers", dtype=dtype).to("cuda")
output = pipe(
    image=Image.open("object.png"),
    num_inference_steps=20,
    guidance_scale=3.0,
    num_gaussians=32768,
    generator=torch.Generator("cuda").manual_seed(42),
    decoder_generator=torch.Generator("cuda").manual_seed(1234),
)
export_to_gaussian_ply(output.gaussians[0], "object.ply")
export_to_splat(output.gaussians[0], "object.splat")
```

Use `num_gaussians=[32768, 65536]` to decode several densities after one denoising run. The output contains one
batched Gaussian tensor per requested density. Counts range from 32768 to 262144 and are rounded to multiples of
32 for the released decoder. Set `output_type="latent"` to return the denoised Gaussian and camera latents.

Tensor image inputs use `(channels, height, width)` or `(batch, channels, height, width)` layout. NumPy image
arrays use channel-last layout. Float image inputs use values between zero and one.

`TripoSplatImageProcessor.prepare_foreground()` crops RGBA foregrounds and returns RGB PIL images on black
canvases. The pipeline predicts missing alpha masks with the optional BiRefNet component before this step.
The processor inherits `VaeImageProcessor.preprocess()` to convert the prepared RGB images to tensors in `[0, 1]`;
the DINO and VAE encoding steps apply their respective normalization.

## Modular pipeline

```python
import torch
from PIL import Image
from diffusers import ModularPipeline

pipe = ModularPipeline.from_pretrained("./triposplat-diffusers")
pipe.load_components(dtype={"default": torch.float16, "image_encoder": torch.bfloat16, "vae": torch.bfloat16})
pipe.to("cuda")
pipe.update_components(guider=pipe.guider.new(guidance_scale=3.0))
state = pipe(
    image=Image.open("object.png"),
    num_inference_steps=20,
    num_gaussians=32768,
    generator=torch.Generator("cuda").manual_seed(42),
    decoder_generator=torch.Generator("cuda").manual_seed(1234),
)
gaussians = state.get("gaussians")
```

The default blockset has `preprocess`, `image_encoder`, `vae_encoder`, `denoise`, and `decode` steps.
The denoise sequence separates conditioning expansion, timestep setup, noise preparation, and the denoising loop.
Each step can be removed or run independently. For example, remove `decode` from a `TripoSplatAutoBlocks`
instance before calling `init_pipeline()` to obtain latents, then reuse them with the decode block at different
densities. Configure guidance on the guider component.

Updated `from_config` components, including `image_processor`, are not stored in the modular index by the current
framework. After reloading, reapply a customized processor with `update_components(image_processor=...)`.
The default 1024-pixel processor is unaffected.

`TripoSplatClassifierFreeGuidance` evaluates guidance as `scale * conditional - (scale - 1) * unconditional`.
The standard guider uses an equivalent formula with a different operation order, which changes float16 rounding.
TripoSplat uses only the conditional prediction for guidance scales at or below one.

## TripoSplatPipeline

[[autodoc]] TripoSplatPipeline
    - all
    - __call__

## TripoSplatPipelineOutput

[[autodoc]] TripoSplatPipelineOutput

## TripoSplatModularPipeline

[[autodoc]] TripoSplatModularPipeline
    - all

## BiRefNetModel

[[autodoc]] BiRefNetModel
    - all
    - forward
