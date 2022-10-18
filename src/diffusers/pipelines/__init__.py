from ..utils import is_flax_available, is_onnx_available, is_torch_available, is_transformers_available


if is_torch_available():
    from .ddim import DDIMPipeline
    from .ddpm import DDPMPipeline
    from .latent_diffusion_uncond import LDMPipeline
    from .pndm import PNDMPipeline
    from .score_sde_ve import ScoreSdeVePipeline
    from .stochastic_karras_ve import KarrasVePipeline
else:
    from ..utils.dummy_pt_objects import *  # noqa F403

if is_torch_available() and is_transformers_available():
    from .latent_diffusion import LDMTextToImagePipeline
    from .stable_diffusion import (
        StableDiffusionImg2ImgPipeline,
        StableDiffusionInpaintPipeline,
        StableDiffusionPipeline,
    )

if is_transformers_available() and is_onnx_available():
    from .stable_diffusion import (
        OnnxStableDiffusionImg2ImgPipeline,
        OnnxStableDiffusionInpaintPipeline,
        OnnxStableDiffusionPipeline,
        StableDiffusionOnnxPipeline,
    )

if is_transformers_available() and is_flax_available():
    from .stable_diffusion import FlaxStableDiffusionPipeline
