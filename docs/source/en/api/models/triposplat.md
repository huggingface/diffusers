# TripoSplat models

[TripoSplat](https://github.com/VAST-AI-Research/TripoSplat) generates 3D Gaussian splats from an image. Its transformer
jointly denoises Gaussian and camera latents, conditioned on DINOv3 features and packed Flux2 VAE image latents.
The Gaussian decoder samples an octree and predicts splat parameters at the requested density.

## TripoSplatTransformer3DModel

[[autodoc]] TripoSplatTransformer3DModel
    - all
    - forward

## TripoSplatGaussianDecoder

[[autodoc]] TripoSplatGaussianDecoder
    - all
    - forward

The decoder returns a tensor of shape `(batch, num_gaussians, 14)`. Columns contain xyz position (3), degree-zero
spherical harmonic color (3), positive scale (3), wxyz rotation (4), and opacity (1). Positions use the model's
coordinate system; export helpers apply the reference viewer transform.

The decoder's adaptive octree uses data-dependent tensor sizes. Compile its repeated transformer blocks with
`decoder.compile_repeated_blocks()`; compiling the whole decoder as a static graph is not supported.

## Original single-file checkpoints

The transformer, Gaussian decoder, and optional `BiRefNetModel` support `from_single_file`. Supply the component's
local Diffusers config explicitly; a default converted Hub repository has not been published yet.

```python
import torch
from diffusers import TripoSplatTransformer3DModel

transformer = TripoSplatTransformer3DModel.from_single_file(
    "./ckpts/diffusion_models/triposplat_fp16.safetensors",
    config="./triposplat-diffusers/transformer",
    dtype=torch.float16,
)
```

Use the same loading method with `TripoSplatGaussianDecoder` and its `decoder` config, or `BiRefNetModel` and its
`background_remover` config. The checkpoint key mappings are shared with the conversion script.
