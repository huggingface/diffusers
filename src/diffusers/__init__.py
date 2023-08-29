__version__ = "0.21.0.dev0"

import os
import importlib
from .file_utils import _BaseLazyModule
from .utils import (
    OptionalDependencyNotAvailable,
    is_flax_available,
    is_k_diffusion_available,
    is_librosa_available,
    is_note_seq_available,
    is_onnx_available,
    is_scipy_available,
    is_torch_available,
    is_torchsde_available,
    is_transformers_available,
)

_import_structure = {
    "configuration_utils": ["ConfigMixin"],
    "utils": [
        "OptionalDependencyNotAvailable",
        "is_flax_available",
        "is_inflect_available",
        "is_invisible_watermark_available",
        "is_k_diffusion_available",
        "is_k_diffusion_version",
        "is_librosa_available",
        "is_note_seq_available",
        "is_onnx_available",
        "is_scipy_available",
        "is_torch_available",
        "is_torchsde_available",
        "is_transformers_available",
        "is_transformers_version",
        "is_unidecode_available",
        "logging",
    ],
    "models": [],
    "optimization": [],
    "pipelines": [],
    "schedulers": [],
    "training_utils": [],
}

try:
    if not is_onnx_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils.dummy_onnx_objects import *  # noqa F403
else:
    from .pipelines import OnnxRuntimeModel

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils.dummy_pt_objects import *  # noqa F403
else:
    _import_structure["models"].extend(
        [
            "AsymmetricAutoencoderKL"
            "AutoencoderKL"
            "AutoencoderTiny"
            "ControlNetModel"
            "ModelMixin"
            "MultiAdapter"
            "PriorTransformer"
            "T2IAdapter"
            "T5FilmDecoder"
            "Transformer2DModel"
            "UNet1DModel"
            "UNet2DConditionModel"
            "UNet2DModel"
            "UNet3DConditionModel"
            "VQModel"
        ]
    )
    _import_structure["optimization"].extend(
        [
            "get_constant_schedule",
            "get_constant_schedule_with_warmup",
            "get_cosine_schedule_with_warmup",
            "get_cosine_with_hard_restarts_schedule_with_warmup",
            "get_linear_schedule_with_warmup",
            "get_polynomial_decay_schedule_with_warmup",
            "get_scheduler",
        ]
    )
    _import_structure["pipelines"].extend(
        [
            "AudioPipelineOutput",
            "AutoPipelineForImage2Image",
            "AutoPipelineForInpainting",
            "AutoPipelineForText2Image",
            "ConsistencyModelPipeline",
            "DanceDiffusionPipeline",
            "DDIMPipeline",
            "DDPMPipeline",
            "DiffusionPipeline",
            "DiTPipeline",
            "ImagePipelineOutput",
            "KarrasVePipeline",
            "LDMPipeline",
            "LDMSuperResolutionPipeline",
            "PNDMPipeline",
            "RePaintPipeline",
            "ScoreSdeVePipeline",
        ]
    )
    _import_structure["schedulers"].extend(
        [
            "CMStochasticIterativeScheduler",
            "DDIMInverseScheduler",
            "DDIMParallelScheduler",
            "DDIMScheduler",
            "DDPMParallelScheduler",
            "DDPMScheduler",
            "DEISMultistepScheduler",
            "DPMSolverMultistepInverseScheduler",
            "DPMSolverMultistepScheduler",
            "DPMSolverSinglestepScheduler",
            "EulerAncestralDiscreteScheduler",
            "EulerDiscreteScheduler",
            "HeunDiscreteScheduler",
            "IPNDMScheduler",
            "KarrasVeScheduler",
            "KDPM2AncestralDiscreteScheduler",
            "KDPM2DiscreteScheduler",
            "PNDMScheduler",
            "RePaintScheduler",
            "SchedulerMixin",
            "ScoreSdeVeScheduler",
            "UnCLIPScheduler",
            "UniPCMultistepScheduler",
            "VQDiffusionScheduler",
        ]
    )
    _import_structure["training_utils"].extend(["EMAModel"])

try:
    if not (is_torch_available() and is_scipy_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils.dummy_torch_and_scipy_objects import *  # noqa F403
else:
    _import_structure["schedulers"].extend(["LMSDiscreteScheduler"])

try:
    if not (is_torch_available() and is_torchsde_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils.dummy_torch_and_torchsde_objects import *  # noqa F403
else:
    _import_structure["schedulers"].extend(["DPMSolverSDEScheduler"])

try:
    if not (is_torch_available() and is_transformers_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils.dummy_torch_and_transformers_objects import *  # noqa F403
else:
    _import_structure["pipelines"].extend(
        [
            "AltDiffusionPipeline",
            "AudioLDM2Pipeline",
            "AudioLDM2ProjectionModel",
            "AudioLDM2UNet2DConditionModel",
            "AudioLDMPipeline",
            "CycleDiffusionPipeline",
            "IFImg2ImgPipeline",
            "IFImg2ImgSuperResolutionPipeline",
            "IFInpaintingPipeline",
            "IFInpaintingSuperResolutionPipeline",
            "IFPipeline",
            "IFSuperResolutionPipeline",
            "ImageTextPipelineOutput",
            "KandinskyCombinedPipeline",
            "KandinskyImg2ImgCombinedPipeline",
            "KandinskyImg2ImgPipeline",
            "KandinskyInpaintCombinedPipeline",
            "KandinskyInpaintPipeline",
            "KandinskyPipeline",
            "KandinskyPriorPipeline",
            "KandinskyV22CombinedPipeline",
            "KandinskyV22ControlnetImg2ImgPipeline",
            "KandinskyV22ControlnetPipeline",
            "KandinskyV22Img2ImgCombinedPipeline",
            "KandinskyV22Img2ImgPipeline",
            "KandinskyV22InpaintCombinedPipeline",
            "KandinskyV22InpaintPipeline",
            "KandinskyV22Pipeline",
            "KandinskyV22PriorEmb2EmbPipeline",
            "KandinskyV22PriorPipeline",
            "LDMTextToImagePipeline",
            "PaintByExamplePipeline",
            "SemanticStableDiffusionPipeline",
            "ShapEImg2ImgPipeline",
            "ShapEPipeline",
            "StableDiffusionAdapterPipeline",
            "StableDiffusionAttendAndExcitePipeline",
            "StableDiffusionControlNetImg2ImgPipeline",
            "StableDiffusionControlNetInpaintPipeline",
            "StableDiffusionControlNetPipeline",
            "StableDiffusionDepth2ImgPipeline",
            "StableDiffusionDiffEditPipeline",
            "StableDiffusionGLIGENPipeline",
            "StableDiffusionImageVariationPipeline",
            "StableDiffusionImg2ImgPipeline",
            "StableDiffusionInpaintPipeline",
            "StableDiffusionInpaintPipelineLegacy",
            "StableDiffusionInstructPix2PixPipeline",
            "StableDiffusionLatentUpscalePipeline",
            "StableDiffusionLDM3DPipeline",
            "StableDiffusionModelEditingPipeline",
            "StableDiffusionPanoramaPipeline",
            "StableDiffusionParadigmsPipeline",
            "StableDiffusionPipeline",
            "StableDiffusionPipelineSafe",
            "StableDiffusionPix2PixZeroPipeline",
            "StableDiffusionSAGPipeline",
            "StableDiffusionUpscalePipeline",
            "StableDiffusionXLControlNetPipeline",
            "StableDiffusionXLImg2ImgPipeline",
            "StableDiffusionXLInpaintPipeline",
            "StableDiffusionXLInstructPix2PixPipeline",
            "StableDiffusionXLPipeline",
            "StableUnCLIPImg2ImgPipeline",
            "StableUnCLIPPipeline",
            "TextToVideoSDPipeline",
            "TextToVideoZeroPipeline",
            "UnCLIPImageVariationPipeline",
            "UnCLIPPipeline",
            "UniDiffuserModel",
            "UniDiffuserPipeline",
            "UniDiffuserTextDecoder",
            "VersatileDiffusionDualGuidedPipeline",
            "VersatileDiffusionImageVariationPipeline",
            "VersatileDiffusionPipeline",
            "VersatileDiffusionTextToImagePipeline",
            "VideoToVideoSDPipeline",
            "VQDiffusionPipeline",
        ]
    )

try:
    if not (is_torch_available() and is_transformers_available() and is_k_diffusion_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils.dummy_torch_and_transformers_and_k_diffusion_objects import *  # noqa F403
else:
    _import_structure["pipelines"].extend(["StableDiffusionKDiffusionPipeline"])

try:
    if not (is_torch_available() and is_transformers_available() and is_onnx_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils.dummy_torch_and_transformers_and_onnx_objects import *  # noqa F403
else:
    _import_structure["pipelines"].extend(
        [
            "OnnxStableDiffusionImg2ImgPipeline",
            "OnnxStableDiffusionInpaintPipeline",
            "OnnxStableDiffusionInpaintPipelineLegacy",
            "OnnxStableDiffusionPipeline",
            "OnnxStableDiffusionUpscalePipeline",
            "StableDiffusionOnnxPipeline",
        ]
    )

try:
    if not (is_torch_available() and is_librosa_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils.dummy_torch_and_librosa_objects import *  # noqa F403
else:
    _import_structure["pipelines"].extend(["AudioDiffusionPipeline", "Mel"])

try:
    if not (is_transformers_available() and is_torch_available() and is_note_seq_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils.dummy_transformers_and_torch_and_note_seq_objects import *  # noqa F403
else:
    _import_structure["pipelines"].extend(["SpectrogramDiffusionPipeline"])

try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils.dummy_flax_objects import *  # noqa F403
else:
    _import_structure["models.controlnet_flax"] = ["FlaxControlNetModel"]
    _import_structure["models.modeling_flax_utils"] = ["FlaxModelMixin"]
    _import_structure["models.unet_2d_condition_flax"] = ["FlaxUNet2DConditionModel"]
    _import_structure["models.vae_flax"] = ["FlaxAutoencoderKL"]
    _import_structure["pipelines"].extend(["FlaxDiffusionPipeline"])
    _import_structure["schedulers"].extend(
        [
            "FlaxDDIMScheduler",
            "FlaxDDPMScheduler",
            "FlaxDPMSolverMultistepScheduler",
            "FlaxKarrasVeScheduler",
            "FlaxLMSDiscreteScheduler",
            "FlaxPNDMScheduler",
            "FlaxSchedulerMixin",
            "FlaxScoreSdeVeScheduler",
        ]
    )


try:
    if not (is_flax_available() and is_transformers_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils.dummy_flax_and_transformers_objects import *  # noqa F403
else:
    _import_structure["pipelines"].extend(
        [
            "FlaxStableDiffusionControlNetPipeline",
            "FlaxStableDiffusionImg2ImgPipeline",
            "FlaxStableDiffusionInpaintPipeline",
            "FlaxStableDiffusionPipeline",
        ]
    )

try:
    if not (is_note_seq_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils.dummy_note_seq_objects import *  # noqa F403
else:
    _import_structure["pipelines"].extend(["MidiProcessor"])

import sys


class _LazyModule(_BaseLazyModule):
    """
    Module class that surfaces all objects but only performs associated imports when the objects are requested.
    """

    __file__ = globals()["__file__"]
    __path__ = [os.path.dirname(__file__)]

    def _get_module(self, module_name: str):
        return importlib.import_module("." + module_name, self.__name__)

    def __getattr__(self, name: str):
        # Special handling for the version, which is a constant from this module and not imported in a submodule.
        if name == "__version__":
            return __version__
        return super().__getattr__(name)


sys.modules[__name__] = _LazyModule(__name__, _import_structure)
