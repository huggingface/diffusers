from typing import TYPE_CHECKING

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


_dummy_objects = {}
_import_structure = {}

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ...utils import dummy_pt_objects

    _dummy_objects.update(get_objects_from_module(dummy_pt_objects))
else:
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
    _import_structure["versatile_diffusion"] = [
        "VersatileDiffusionDualGuidedPipeline",
        "VersatileDiffusionImageVariationPipeline",
        "VersatileDiffusionPipeline",
        "VersatileDiffusionTextToImagePipeline",
    ]
    _import_structure["vq_diffusion"] = ["VQDiffusionPipeline"]
    _import_structure["stable_diffusion_variants"] = [
        "CycleDiffusionPipeline",
        "StableDiffusionInpaintPipelineLegacy",
        "StableDiffusionPix2PixZeroPipeline",
        "StableDiffusionParadigmsPipeline",
        "StableDiffusionModelEditingPipeline",
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
        from .audio_diffusion import AudioDiffusionPipeline, Mel
        from .spectrogram_diffusion import SpectrogramDiffusionPipeline
        from .stable_diffusion_variants import (
            CycleDiffusionPipeline,
            StableDiffusionInpaintPipelineLegacy,
            StableDiffusionModelEditingPipeline,
            StableDiffusionParadigmsPipeline,
            StableDiffusionPix2PixZeroPipeline,
        )
        from .stochastic_karras_ve import KarrasVePipeline
        from .versatile_diffusion import (
            VersatileDiffusionDualGuidedPipeline,
            VersatileDiffusionImageVariationPipeline,
            VersatileDiffusionPipeline,
            VersatileDiffusionTextToImagePipeline,
        )
        from .vq_diffusion import VQDiffusionPipeline

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

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )
    for name, value in _dummy_objects.items():
        setattr(sys.modules[__name__], name, value)
