from typing import TYPE_CHECKING
import functools
import inspect
import warnings
import sys

from ...utils import (
    DIFFUSERS_SLOW_IMPORT,
    OptionalDependencyNotAvailable,
    _LazyModule,
    get_objects_from_module,
    is_librosa_available,
    is_note_seq_available,
    is_torch_available,
    is_transformers_available,
)


# Custom Lazy Module for deprecated pipelines that shows a warning
class _DeprecatedLazyModule(_LazyModule):
    """
    Module class that surfaces all objects but only performs associated imports when the objects are requested,
    and shows deprecation warnings when any of its attributes are accessed.
    """

    def __getattr__(self, name):
        # Regular attribute access - first check if it's supposed to be loaded
        if name in self._modules or name in self._class_to_module:
            # Only warn for actual pipeline components, not utility functions
            warnings.warn(
                f"{name} is deprecated and will no longer be maintained.",
                FutureWarning,
                stacklevel=2,
            )

        # Use the standard lazy module behavior to load the attribute
        return super().__getattr__(name)


_dummy_objects = {}
_import_structure = {}

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ...utils import dummy_pt_objects

    _dummy_objects.update(get_objects_from_module(dummy_pt_objects))
else:
    _import_structure["dance_diffusion"] = ["DanceDiffusionPipeline"]
    _import_structure["latent_diffusion_uncond"] = ["LDMPipeline"]
    _import_structure["pndm"] = ["PNDMPipeline"]
    _import_structure["repaint"] = ["RePaintPipeline"]
    _import_structure["score_sde_ve"] = ["ScoreSdeVePipeline"]
    _import_structure["stochastic_karras_ve"] = ["KarrasVePipeline"]

try:
    if not (is_transformers_available() and is_torch_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ...utils import dummy_torch_and_transformers_objects

    _dummy_objects.update(get_objects_from_module(dummy_torch_and_transformers_objects))
else:
    _import_structure["alt_diffusion"] = [
        "AltDiffusionImg2ImgPipeline",
        "AltDiffusionPipeline",
        "AltDiffusionPipelineOutput",
    ]
    _import_structure["amused"] = ["AmusedPipeline", "AmusedImg2ImgPipeline", "AmusedInpaintPipeline"]
    _import_structure["audioldm"] = ["AudioLDMPipeline"]
    _import_structure["controlnet_xs"] = [
        "StableDiffusionControlNetXSPipeline",
        "StableDiffusionXLControlNetXSPipeline",
    ]
    _import_structure["i2vgen_xl"] = ["I2VGenXLPipeline"]
    _import_structure["latte"] = ["LattePipeline"]
    _import_structure["musicldm"] = ["MusicLDMPipeline"]
    _import_structure["paint_by_example"] = ["PaintByExamplePipeline"]
    _import_structure["pia"] = ["PIAPipeline"]
    _import_structure["semantic_stable_diffusion"] = ["SemanticStableDiffusionPipeline"]
    _import_structure["shap_e"] = [
        "ShapEPipeline",
        "ShapEImg2ImgPipeline",
    ]
    _import_structure["stable_diffusion_attend_and_excite"] = ["StableDiffusionAttendAndExcitePipeline"]
    _import_structure["stable_diffusion_diffedit"] = ["StableDiffusionDiffEditPipeline"]
    _import_structure["stable_diffusion_gligen"] = [
        "StableDiffusionGLIGENPipeline",
        "StableDiffusionGLIGENTextImagePipeline",
    ]
    _import_structure["stable_diffusion_k_diffusion"] = [
        "StableDiffusionKDiffusionPipeline",
        "StableDiffusionXLKDiffusionPipeline",
    ]
    _import_structure["stable_diffusion_ldm3d"] = ["StableDiffusionLDM3DPipeline"]
    _import_structure["stable_diffusion_panorama"] = ["StableDiffusionPanoramaPipeline"]
    _import_structure["stable_diffusion_safe"] = ["StableDiffusionPipelineSafe"]
    _import_structure["stable_diffusion_sag"] = ["StableDiffusionSAGPipeline"]
    _import_structure["stable_diffusion_variants"] = [
        "CycleDiffusionPipeline",
        "StableDiffusionInpaintPipelineLegacy",
        "StableDiffusionPix2PixZeroPipeline",
        "StableDiffusionParadigmsPipeline",
        "StableDiffusionModelEditingPipeline",
    ]
    _import_structure["text_to_video_synthesis"] = [
        "TextToVideoSDPipeline",
        "TextToVideoZeroPipeline",
        "TextToVideoZeroSDXLPipeline",
    ]
    _import_structure["unclip"] = [
        "UnCLIPPipeline",
        "UnCLIPImageVariationPipeline",
    ]
    _import_structure["unidiffuser"] = [
        "UniDiffuserPipeline",
    ]
    _import_structure["versatile_diffusion"] = [
        "VersatileDiffusionDualGuidedPipeline",
        "VersatileDiffusionImageVariationPipeline",
        "VersatileDiffusionPipeline",
        "VersatileDiffusionTextToImagePipeline",
    ]
    _import_structure["vq_diffusion"] = ["VQDiffusionPipeline"]
    _import_structure["wuerstchen"] = [
        "WuerstchenCombinedPipeline",
        "WuerstchenDecoderPipeline",
        "WuerstchenPriorPipeline",
    ]

try:
    if not (is_torch_available() and is_librosa_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ...utils import dummy_torch_and_librosa_objects  # noqa F403

    _dummy_objects.update(get_objects_from_module(dummy_torch_and_librosa_objects))

else:
    _import_structure["audio_diffusion"] = ["AudioDiffusionPipeline", "Mel"]

try:
    if not (is_transformers_available() and is_torch_available() and is_note_seq_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ...utils import dummy_transformers_and_torch_and_note_seq_objects  # noqa F403

    _dummy_objects.update(get_objects_from_module(dummy_transformers_and_torch_and_note_seq_objects))

else:
    _import_structure["spectrogram_diffusion"] = ["MidiProcessor", "SpectrogramDiffusionPipeline"]


if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from ...utils.dummy_pt_objects import *

    else:
        from .dance_diffusion import DanceDiffusionPipeline
        from .latent_diffusion_uncond import LDMPipeline
        from .pndm import PNDMPipeline
        from .repaint import RePaintPipeline
        from .score_sde_ve import ScoreSdeVePipeline
        from .stochastic_karras_ve import KarrasVePipeline

    try:
        if not (is_transformers_available() and is_torch_available()):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from ...utils.dummy_torch_and_transformers_objects import *

    else:
        from .alt_diffusion import AltDiffusionImg2ImgPipeline, AltDiffusionPipeline, AltDiffusionPipelineOutput
        from .audioldm import AudioLDMPipeline
        from .controlnet_xs import StableDiffusionControlNetXSPipeline, StableDiffusionXLControlNetXSPipeline
        from .i2vgen_xl import I2VGenXLPipeline
        from .latte import LattePipeline
        from .musicldm import MusicLDMPipeline
        from .paint_by_example import PaintByExamplePipeline
        from .pia import PIAPipeline
        from .semantic_stable_diffusion import SemanticStableDiffusionPipeline
        from .shap_e import ShapEPipeline, ShapEImg2ImgPipeline
        from .stable_diffusion_attend_and_excite import StableDiffusionAttendAndExcitePipeline
        from .stable_diffusion_diffedit import StableDiffusionDiffEditPipeline
        from .stable_diffusion_gligen import StableDiffusionGLIGENPipeline, StableDiffusionGLIGENTextImagePipeline
        from .stable_diffusion_k_diffusion import (
            StableDiffusionKDiffusionPipeline,
            StableDiffusionXLKDiffusionPipeline,
        )
        from .stable_diffusion_ldm3d import StableDiffusionLDM3DPipeline
        from .stable_diffusion_panorama import StableDiffusionPanoramaPipeline
        from .stable_diffusion_safe import StableDiffusionPipelineSafe
        from .stable_diffusion_sag import StableDiffusionSAGPipeline
        from .stable_diffusion_variants import (
            CycleDiffusionPipeline,
            StableDiffusionInpaintPipelineLegacy,
            StableDiffusionModelEditingPipeline,
            StableDiffusionParadigmsPipeline,
            StableDiffusionPix2PixZeroPipeline,
        )
        from .text_to_video_synthesis import (
            TextToVideoSDPipeline,
            TextToVideoZeroPipeline,
            TextToVideoZeroSDXLPipeline,
        )
        from .unclip import UnCLIPPipeline, UnCLIPImageVariationPipeline
        from .unidiffuser import UniDiffuserPipeline
        from .versatile_diffusion import (
            VersatileDiffusionDualGuidedPipeline,
            VersatileDiffusionImageVariationPipeline,
            VersatileDiffusionPipeline,
            VersatileDiffusionTextToImagePipeline,
        )
        from .vq_diffusion import VQDiffusionPipeline
        from .wuerstchen import WuerstchenCombinedPipeline, WuerstchenDecoderPipeline, WuerstchenPriorPipeline

    try:
        if not (is_torch_available() and is_librosa_available()):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from ...utils.dummy_torch_and_librosa_objects import *
    else:
        from .audio_diffusion import AudioDiffusionPipeline, Mel

    try:
        if not (is_transformers_available() and is_torch_available() and is_note_seq_available()):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from ...utils.dummy_transformers_and_torch_and_note_seq_objects import *  # noqa F403
    else:
        from .spectrogram_diffusion import (
            MidiProcessor,
            SpectrogramDiffusionPipeline,
        )


else:
    import sys

    # Use the custom deprecated lazy module instead of the standard one
    sys.modules[__name__] = _DeprecatedLazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )
    for name, value in _dummy_objects.items():
        setattr(sys.modules[__name__], name, value)
