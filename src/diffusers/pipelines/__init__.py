from ..utils import (
    OptionalDependencyNotAvailable,
    is_flax_available,
    is_k_diffusion_available,
    is_librosa_available,
    is_onnx_available,
    is_torch_available,
    is_transformers_available,
)


try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ..utils.dummy_pt_objects import *  # noqa F403
else:
    from .dance_diffusion import DanceDiffusionPipeline
    from .ddim import DDIMPipeline
    from .ddpm import DDPMPipeline
    from .latent_diffusion import LDMSuperResolutionPipeline
    from .latent_diffusion_uncond import LDMPipeline
    from .pndm import PNDMPipeline
    from .repaint import RePaintPipeline
    from .score_sde_ve import ScoreSdeVePipeline
    from .stochastic_karras_ve import KarrasVePipeline

try:
    if not (is_torch_available() and is_librosa_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ..utils.dummy_torch_and_librosa_objects import *  # noqa F403
else:
    from .audio_diffusion import AudioDiffusionPipeline, Mel

try:
    if not (is_torch_available() and is_transformers_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ..utils.dummy_torch_and_transformers_objects import *  # noqa F403
else:
    from .alt_diffusion import AltDiffusionImg2ImgPipeline, AltDiffusionPipeline
    from .latent_diffusion import LDMTextToImagePipeline
    from .paint_by_example import PaintByExamplePipeline
    from .stable_diffusion import (
        CycleDiffusionPipeline,
        StableDiffusionDepth2ImgPipeline,
        StableDiffusionImageVariationPipeline,
        StableDiffusionImg2ImgPipeline,
        StableDiffusionInpaintPipeline,
        StableDiffusionInpaintPipelineLegacy,
        StableDiffusionPipeline,
        StableDiffusionUpscalePipeline,
    )
    from .stable_diffusion_safe import StableDiffusionPipelineSafe
    from .unclip import UnCLIPImageVariationPipeline, UnCLIPPipeline
    from .versatile_diffusion import (
        VersatileDiffusionDualGuidedPipeline,
        VersatileDiffusionImageVariationPipeline,
        VersatileDiffusionPipeline,
        VersatileDiffusionTextToImagePipeline,
    )
    from .vq_diffusion import VQDiffusionPipeline

try:
    if not (is_torch_available() and is_transformers_available() and is_onnx_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ..utils.dummy_torch_and_transformers_and_onnx_objects import *  # noqa F403
else:
    from .stable_diffusion import (
        OnnxStableDiffusionImg2ImgPipeline,
        OnnxStableDiffusionInpaintPipeline,
        OnnxStableDiffusionInpaintPipelineLegacy,
        OnnxStableDiffusionPipeline,
        StableDiffusionOnnxPipeline,
    )

try:
    if not (is_torch_available() and is_transformers_available() and is_k_diffusion_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ..utils.dummy_torch_and_transformers_and_k_diffusion_objects import *  # noqa F403
else:
    from .stable_diffusion import StableDiffusionKDiffusionPipeline


try:
    if not (is_flax_available() and is_transformers_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ..utils.dummy_flax_and_transformers_objects import *  # noqa F403
else:
    from .stable_diffusion import FlaxStableDiffusionImg2ImgPipeline, FlaxStableDiffusionPipeline
