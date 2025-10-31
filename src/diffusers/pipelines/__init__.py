from typing import TYPE_CHECKING

from ..utils import (
    DIFFUSERS_SLOW_IMPORT,
    OptionalDependencyNotAvailable,
    _LazyModule,
    get_objects_from_module,
    is_flax_available,
    is_k_diffusion_available,
    is_librosa_available,
    is_note_seq_available,
    is_onnx_available,
    is_opencv_available,
    is_sentencepiece_available,
    is_torch_available,
    is_torch_npu_available,
    is_transformers_available,
)


# These modules contain pipelines from multiple libraries/frameworks
_dummy_objects = {}
_import_structure = {
    "controlnet": [],
    "controlnet_hunyuandit": [],
    "controlnet_sd3": [],
    "controlnet_xs": [],
    "deprecated": [],
    "latent_diffusion": [],
    "ledits_pp": [],
    "marigold": [],
    "pag": [],
    "stable_diffusion": [],
    "stable_diffusion_xl": [],
}

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ..utils import dummy_pt_objects  # noqa F403

    _dummy_objects.update(get_objects_from_module(dummy_pt_objects))
else:
    _import_structure["auto_pipeline"] = [
        "AutoPipelineForImage2Image",
        "AutoPipelineForInpainting",
        "AutoPipelineForText2Image",
    ]
    _import_structure["consistency_models"] = ["ConsistencyModelPipeline"]
    _import_structure["dance_diffusion"] = ["DanceDiffusionPipeline"]
    _import_structure["ddim"] = ["DDIMPipeline"]
    _import_structure["ddpm"] = ["DDPMPipeline"]
    _import_structure["dit"] = ["DiTPipeline"]
    _import_structure["latent_diffusion"].extend(["LDMSuperResolutionPipeline"])
    _import_structure["pipeline_utils"] = [
        "AudioPipelineOutput",
        "DiffusionPipeline",
        "StableDiffusionMixin",
        "ImagePipelineOutput",
    ]
    _import_structure["deprecated"].extend(
        [
            "PNDMPipeline",
            "LDMPipeline",
            "RePaintPipeline",
            "ScoreSdeVePipeline",
            "KarrasVePipeline",
        ]
    )
try:
    if not (is_torch_available() and is_librosa_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ..utils import dummy_torch_and_librosa_objects  # noqa F403

    _dummy_objects.update(get_objects_from_module(dummy_torch_and_librosa_objects))
else:
    _import_structure["deprecated"].extend(["AudioDiffusionPipeline", "Mel"])

try:
    if not (is_transformers_available() and is_torch_available() and is_note_seq_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ..utils import dummy_transformers_and_torch_and_note_seq_objects  # noqa F403

    _dummy_objects.update(get_objects_from_module(dummy_transformers_and_torch_and_note_seq_objects))
else:
    _import_structure["deprecated"].extend(
        [
            "MidiProcessor",
            "SpectrogramDiffusionPipeline",
        ]
    )

try:
    if not (is_torch_available() and is_transformers_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ..utils import dummy_torch_and_transformers_objects  # noqa F403

    _dummy_objects.update(get_objects_from_module(dummy_torch_and_transformers_objects))
else:
    _import_structure["deprecated"].extend(
        [
            "VQDiffusionPipeline",
            "AltDiffusionPipeline",
            "AltDiffusionImg2ImgPipeline",
            "CycleDiffusionPipeline",
            "StableDiffusionInpaintPipelineLegacy",
            "StableDiffusionPix2PixZeroPipeline",
            "StableDiffusionParadigmsPipeline",
            "StableDiffusionModelEditingPipeline",
            "VersatileDiffusionDualGuidedPipeline",
            "VersatileDiffusionImageVariationPipeline",
            "VersatileDiffusionPipeline",
            "VersatileDiffusionTextToImagePipeline",
        ]
    )
    _import_structure["allegro"] = ["AllegroPipeline"]
    _import_structure["amused"] = ["AmusedImg2ImgPipeline", "AmusedInpaintPipeline", "AmusedPipeline"]
    _import_structure["animatediff"] = [
        "AnimateDiffPipeline",
        "AnimateDiffControlNetPipeline",
        "AnimateDiffSDXLPipeline",
        "AnimateDiffSparseControlNetPipeline",
        "AnimateDiffVideoToVideoPipeline",
        "AnimateDiffVideoToVideoControlNetPipeline",
    ]
    _import_structure["bria"] = ["BriaPipeline"]
    _import_structure["bria_fibo"] = ["BriaFiboPipeline"]
    _import_structure["flux"] = [
        "FluxControlPipeline",
        "FluxControlInpaintPipeline",
        "FluxControlImg2ImgPipeline",
        "FluxControlNetPipeline",
        "FluxControlNetImg2ImgPipeline",
        "FluxControlNetInpaintPipeline",
        "FluxImg2ImgPipeline",
        "FluxInpaintPipeline",
        "FluxPipeline",
        "FluxFillPipeline",
        "FluxPriorReduxPipeline",
        "ReduxImageEncoder",
        "FluxKontextPipeline",
        "FluxKontextInpaintPipeline",
    ]
    _import_structure["prx"] = ["PRXPipeline"]
    _import_structure["audioldm"] = ["AudioLDMPipeline"]
    _import_structure["audioldm2"] = [
        "AudioLDM2Pipeline",
        "AudioLDM2ProjectionModel",
        "AudioLDM2UNet2DConditionModel",
    ]
    _import_structure["blip_diffusion"] = ["BlipDiffusionPipeline"]
    _import_structure["chroma"] = ["ChromaPipeline", "ChromaImg2ImgPipeline"]
    _import_structure["cogvideo"] = [
        "CogVideoXPipeline",
        "CogVideoXImageToVideoPipeline",
        "CogVideoXVideoToVideoPipeline",
        "CogVideoXFunControlPipeline",
    ]
    _import_structure["cogview3"] = ["CogView3PlusPipeline"]
    _import_structure["cogview4"] = ["CogView4Pipeline", "CogView4ControlPipeline"]
    _import_structure["consisid"] = ["ConsisIDPipeline"]
    _import_structure["cosmos"] = [
        "Cosmos2TextToImagePipeline",
        "CosmosTextToWorldPipeline",
        "CosmosVideoToWorldPipeline",
        "Cosmos2VideoToWorldPipeline",
    ]
    _import_structure["controlnet"].extend(
        [
            "BlipDiffusionControlNetPipeline",
            "StableDiffusionControlNetImg2ImgPipeline",
            "StableDiffusionControlNetInpaintPipeline",
            "StableDiffusionControlNetPipeline",
            "StableDiffusionXLControlNetImg2ImgPipeline",
            "StableDiffusionXLControlNetInpaintPipeline",
            "StableDiffusionXLControlNetPipeline",
            "StableDiffusionXLControlNetUnionPipeline",
            "StableDiffusionXLControlNetUnionInpaintPipeline",
            "StableDiffusionXLControlNetUnionImg2ImgPipeline",
        ]
    )
    _import_structure["pag"].extend(
        [
            "StableDiffusionControlNetPAGInpaintPipeline",
            "AnimateDiffPAGPipeline",
            "KolorsPAGPipeline",
            "HunyuanDiTPAGPipeline",
            "StableDiffusion3PAGPipeline",
            "StableDiffusion3PAGImg2ImgPipeline",
            "StableDiffusionPAGPipeline",
            "StableDiffusionPAGImg2ImgPipeline",
            "StableDiffusionPAGInpaintPipeline",
            "StableDiffusionControlNetPAGPipeline",
            "StableDiffusionXLPAGPipeline",
            "StableDiffusionXLPAGInpaintPipeline",
            "StableDiffusionXLControlNetPAGImg2ImgPipeline",
            "StableDiffusionXLControlNetPAGPipeline",
            "StableDiffusionXLPAGImg2ImgPipeline",
            "PixArtSigmaPAGPipeline",
            "SanaPAGPipeline",
        ]
    )
    _import_structure["controlnet_xs"].extend(
        [
            "StableDiffusionControlNetXSPipeline",
            "StableDiffusionXLControlNetXSPipeline",
        ]
    )
    _import_structure["controlnet_hunyuandit"].extend(
        [
            "HunyuanDiTControlNetPipeline",
        ]
    )
    _import_structure["controlnet_sd3"].extend(
        [
            "StableDiffusion3ControlNetPipeline",
            "StableDiffusion3ControlNetInpaintingPipeline",
        ]
    )
    _import_structure["deepfloyd_if"] = [
        "IFImg2ImgPipeline",
        "IFImg2ImgSuperResolutionPipeline",
        "IFInpaintingPipeline",
        "IFInpaintingSuperResolutionPipeline",
        "IFPipeline",
        "IFSuperResolutionPipeline",
    ]
    _import_structure["easyanimate"] = [
        "EasyAnimatePipeline",
        "EasyAnimateInpaintPipeline",
        "EasyAnimateControlPipeline",
    ]
    _import_structure["hidream_image"] = ["HiDreamImagePipeline"]
    _import_structure["hunyuandit"] = ["HunyuanDiTPipeline"]
    _import_structure["hunyuan_video"] = [
        "HunyuanVideoPipeline",
        "HunyuanSkyreelsImageToVideoPipeline",
        "HunyuanVideoImageToVideoPipeline",
        "HunyuanVideoFramepackPipeline",
    ]
    _import_structure["hunyuan_image"] = ["HunyuanImagePipeline", "HunyuanImageRefinerPipeline"]
    _import_structure["kandinsky"] = [
        "KandinskyCombinedPipeline",
        "KandinskyImg2ImgCombinedPipeline",
        "KandinskyImg2ImgPipeline",
        "KandinskyInpaintCombinedPipeline",
        "KandinskyInpaintPipeline",
        "KandinskyPipeline",
        "KandinskyPriorPipeline",
    ]
    _import_structure["kandinsky2_2"] = [
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
    ]
    _import_structure["kandinsky3"] = [
        "Kandinsky3Img2ImgPipeline",
        "Kandinsky3Pipeline",
    ]
    _import_structure["latent_consistency_models"] = [
        "LatentConsistencyModelImg2ImgPipeline",
        "LatentConsistencyModelPipeline",
    ]
    _import_structure["latent_diffusion"].extend(["LDMTextToImagePipeline"])
    _import_structure["ledits_pp"].extend(
        [
            "LEditsPPPipelineStableDiffusion",
            "LEditsPPPipelineStableDiffusionXL",
        ]
    )
    _import_structure["latte"] = ["LattePipeline"]
    _import_structure["ltx"] = [
        "LTXPipeline",
        "LTXImageToVideoPipeline",
        "LTXConditionPipeline",
        "LTXLatentUpsamplePipeline",
    ]
    _import_structure["lumina"] = ["LuminaPipeline", "LuminaText2ImgPipeline"]
    _import_structure["lumina2"] = ["Lumina2Pipeline", "Lumina2Text2ImgPipeline"]
    _import_structure["lucy"] = ["LucyEditPipeline"]
    _import_structure["marigold"].extend(
        [
            "MarigoldDepthPipeline",
            "MarigoldIntrinsicsPipeline",
            "MarigoldNormalsPipeline",
        ]
    )
    _import_structure["mochi"] = ["MochiPipeline"]
    _import_structure["musicldm"] = ["MusicLDMPipeline"]
    _import_structure["omnigen"] = ["OmniGenPipeline"]
    _import_structure["visualcloze"] = ["VisualClozePipeline", "VisualClozeGenerationPipeline"]
    _import_structure["paint_by_example"] = ["PaintByExamplePipeline"]
    _import_structure["pia"] = ["PIAPipeline"]
    _import_structure["pixart_alpha"] = ["PixArtAlphaPipeline", "PixArtSigmaPipeline"]
    _import_structure["sana"] = [
        "SanaPipeline",
        "SanaSprintPipeline",
        "SanaControlNetPipeline",
        "SanaSprintImg2ImgPipeline",
    ]
    _import_structure["semantic_stable_diffusion"] = ["SemanticStableDiffusionPipeline"]
    _import_structure["shap_e"] = ["ShapEImg2ImgPipeline", "ShapEPipeline"]
    _import_structure["stable_audio"] = [
        "StableAudioProjectionModel",
        "StableAudioPipeline",
    ]
    _import_structure["stable_cascade"] = [
        "StableCascadeCombinedPipeline",
        "StableCascadeDecoderPipeline",
        "StableCascadePriorPipeline",
    ]
    _import_structure["stable_diffusion"].extend(
        [
            "CLIPImageProjection",
            "StableDiffusionDepth2ImgPipeline",
            "StableDiffusionImageVariationPipeline",
            "StableDiffusionImg2ImgPipeline",
            "StableDiffusionInpaintPipeline",
            "StableDiffusionInstructPix2PixPipeline",
            "StableDiffusionLatentUpscalePipeline",
            "StableDiffusionPipeline",
            "StableDiffusionUpscalePipeline",
            "StableUnCLIPImg2ImgPipeline",
            "StableUnCLIPPipeline",
            "StableDiffusionLDM3DPipeline",
        ]
    )
    _import_structure["aura_flow"] = ["AuraFlowPipeline"]
    _import_structure["stable_diffusion_3"] = [
        "StableDiffusion3Pipeline",
        "StableDiffusion3Img2ImgPipeline",
        "StableDiffusion3InpaintPipeline",
    ]
    _import_structure["stable_diffusion_attend_and_excite"] = ["StableDiffusionAttendAndExcitePipeline"]
    _import_structure["stable_diffusion_safe"] = ["StableDiffusionPipelineSafe"]
    _import_structure["stable_diffusion_sag"] = ["StableDiffusionSAGPipeline"]
    _import_structure["stable_diffusion_gligen"] = [
        "StableDiffusionGLIGENPipeline",
        "StableDiffusionGLIGENTextImagePipeline",
    ]
    _import_structure["stable_video_diffusion"] = ["StableVideoDiffusionPipeline"]
    _import_structure["stable_diffusion_xl"].extend(
        [
            "StableDiffusionXLImg2ImgPipeline",
            "StableDiffusionXLInpaintPipeline",
            "StableDiffusionXLInstructPix2PixPipeline",
            "StableDiffusionXLPipeline",
        ]
    )
    _import_structure["stable_diffusion_diffedit"] = ["StableDiffusionDiffEditPipeline"]
    _import_structure["stable_diffusion_ldm3d"] = ["StableDiffusionLDM3DPipeline"]
    _import_structure["stable_diffusion_panorama"] = ["StableDiffusionPanoramaPipeline"]
    _import_structure["t2i_adapter"] = [
        "StableDiffusionAdapterPipeline",
        "StableDiffusionXLAdapterPipeline",
    ]
    _import_structure["text_to_video_synthesis"] = [
        "TextToVideoSDPipeline",
        "TextToVideoZeroPipeline",
        "TextToVideoZeroSDXLPipeline",
        "VideoToVideoSDPipeline",
    ]
    _import_structure["i2vgen_xl"] = ["I2VGenXLPipeline"]
    _import_structure["unclip"] = ["UnCLIPImageVariationPipeline", "UnCLIPPipeline"]
    _import_structure["unidiffuser"] = [
        "ImageTextPipelineOutput",
        "UniDiffuserModel",
        "UniDiffuserPipeline",
        "UniDiffuserTextDecoder",
    ]
    _import_structure["wuerstchen"] = [
        "WuerstchenCombinedPipeline",
        "WuerstchenDecoderPipeline",
        "WuerstchenPriorPipeline",
    ]
    _import_structure["wan"] = ["WanPipeline", "WanImageToVideoPipeline", "WanVideoToVideoPipeline", "WanVACEPipeline"]
    _import_structure["kandinsky5"] = ["Kandinsky5T2VPipeline"]
    _import_structure["skyreels_v2"] = [
        "SkyReelsV2DiffusionForcingPipeline",
        "SkyReelsV2DiffusionForcingImageToVideoPipeline",
        "SkyReelsV2DiffusionForcingVideoToVideoPipeline",
        "SkyReelsV2ImageToVideoPipeline",
        "SkyReelsV2Pipeline",
    ]
    _import_structure["qwenimage"] = [
        "QwenImagePipeline",
        "QwenImageImg2ImgPipeline",
        "QwenImageInpaintPipeline",
        "QwenImageEditPipeline",
        "QwenImageEditPlusPipeline",
        "QwenImageEditInpaintPipeline",
        "QwenImageControlNetInpaintPipeline",
        "QwenImageControlNetPipeline",
    ]
try:
    if not is_onnx_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ..utils import dummy_onnx_objects  # noqa F403

    _dummy_objects.update(get_objects_from_module(dummy_onnx_objects))
else:
    _import_structure["onnx_utils"] = ["OnnxRuntimeModel"]
try:
    if not (is_torch_available() and is_transformers_available() and is_onnx_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ..utils import dummy_torch_and_transformers_and_onnx_objects  # noqa F403

    _dummy_objects.update(get_objects_from_module(dummy_torch_and_transformers_and_onnx_objects))
else:
    _import_structure["stable_diffusion"].extend(
        [
            "OnnxStableDiffusionImg2ImgPipeline",
            "OnnxStableDiffusionInpaintPipeline",
            "OnnxStableDiffusionPipeline",
            "OnnxStableDiffusionUpscalePipeline",
            "StableDiffusionOnnxPipeline",
        ]
    )

try:
    if not (is_torch_available() and is_transformers_available() and is_k_diffusion_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ..utils import (
        dummy_torch_and_transformers_and_k_diffusion_objects,
    )

    _dummy_objects.update(get_objects_from_module(dummy_torch_and_transformers_and_k_diffusion_objects))
else:
    _import_structure["stable_diffusion_k_diffusion"] = [
        "StableDiffusionKDiffusionPipeline",
        "StableDiffusionXLKDiffusionPipeline",
    ]

try:
    if not (is_torch_available() and is_transformers_available() and is_sentencepiece_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ..utils import (
        dummy_torch_and_transformers_and_sentencepiece_objects,
    )

    _dummy_objects.update(get_objects_from_module(dummy_torch_and_transformers_and_sentencepiece_objects))
else:
    _import_structure["kolors"] = [
        "KolorsPipeline",
        "KolorsImg2ImgPipeline",
    ]

try:
    if not (is_torch_available() and is_transformers_available() and is_opencv_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ..utils import (
        dummy_torch_and_transformers_and_opencv_objects,
    )

    _dummy_objects.update(get_objects_from_module(dummy_torch_and_transformers_and_opencv_objects))
else:
    _import_structure["consisid"] = ["ConsisIDPipeline"]

try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ..utils import dummy_flax_objects  # noqa F403

    _dummy_objects.update(get_objects_from_module(dummy_flax_objects))
else:
    _import_structure["pipeline_flax_utils"] = ["FlaxDiffusionPipeline"]
try:
    if not (is_flax_available() and is_transformers_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ..utils import dummy_flax_and_transformers_objects  # noqa F403

    _dummy_objects.update(get_objects_from_module(dummy_flax_and_transformers_objects))
else:
    _import_structure["controlnet"].extend(["FlaxStableDiffusionControlNetPipeline"])
    _import_structure["stable_diffusion"].extend(
        [
            "FlaxStableDiffusionImg2ImgPipeline",
            "FlaxStableDiffusionInpaintPipeline",
            "FlaxStableDiffusionPipeline",
        ]
    )
    _import_structure["stable_diffusion_xl"].extend(
        [
            "FlaxStableDiffusionXLPipeline",
        ]
    )

if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from ..utils.dummy_pt_objects import *  # noqa F403

    else:
        from .auto_pipeline import (
            AutoPipelineForImage2Image,
            AutoPipelineForInpainting,
            AutoPipelineForText2Image,
        )
        from .consistency_models import ConsistencyModelPipeline
        from .dance_diffusion import DanceDiffusionPipeline
        from .ddim import DDIMPipeline
        from .ddpm import DDPMPipeline
        from .deprecated import KarrasVePipeline, LDMPipeline, PNDMPipeline, RePaintPipeline, ScoreSdeVePipeline
        from .dit import DiTPipeline
        from .latent_diffusion import LDMSuperResolutionPipeline
        from .pipeline_utils import (
            AudioPipelineOutput,
            DiffusionPipeline,
            ImagePipelineOutput,
            StableDiffusionMixin,
        )

    try:
        if not (is_torch_available() and is_librosa_available()):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from ..utils.dummy_torch_and_librosa_objects import *
    else:
        from .deprecated import AudioDiffusionPipeline, Mel

    try:
        if not (is_torch_available() and is_transformers_available()):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from ..utils.dummy_torch_and_transformers_objects import *
    else:
        from .allegro import AllegroPipeline
        from .amused import AmusedImg2ImgPipeline, AmusedInpaintPipeline, AmusedPipeline
        from .animatediff import (
            AnimateDiffControlNetPipeline,
            AnimateDiffPipeline,
            AnimateDiffSDXLPipeline,
            AnimateDiffSparseControlNetPipeline,
            AnimateDiffVideoToVideoControlNetPipeline,
            AnimateDiffVideoToVideoPipeline,
        )
        from .audioldm import AudioLDMPipeline
        from .audioldm2 import (
            AudioLDM2Pipeline,
            AudioLDM2ProjectionModel,
            AudioLDM2UNet2DConditionModel,
        )
        from .aura_flow import AuraFlowPipeline
        from .blip_diffusion import BlipDiffusionPipeline
        from .bria import BriaPipeline
        from .bria_fibo import BriaFiboPipeline
        from .chroma import ChromaImg2ImgPipeline, ChromaPipeline
        from .cogvideo import (
            CogVideoXFunControlPipeline,
            CogVideoXImageToVideoPipeline,
            CogVideoXPipeline,
            CogVideoXVideoToVideoPipeline,
        )
        from .cogview3 import CogView3PlusPipeline
        from .cogview4 import CogView4ControlPipeline, CogView4Pipeline
        from .controlnet import (
            BlipDiffusionControlNetPipeline,
            StableDiffusionControlNetImg2ImgPipeline,
            StableDiffusionControlNetInpaintPipeline,
            StableDiffusionControlNetPipeline,
            StableDiffusionXLControlNetImg2ImgPipeline,
            StableDiffusionXLControlNetInpaintPipeline,
            StableDiffusionXLControlNetPipeline,
            StableDiffusionXLControlNetUnionImg2ImgPipeline,
            StableDiffusionXLControlNetUnionInpaintPipeline,
            StableDiffusionXLControlNetUnionPipeline,
        )
        from .controlnet_hunyuandit import (
            HunyuanDiTControlNetPipeline,
        )
        from .controlnet_sd3 import StableDiffusion3ControlNetInpaintingPipeline, StableDiffusion3ControlNetPipeline
        from .controlnet_xs import (
            StableDiffusionControlNetXSPipeline,
            StableDiffusionXLControlNetXSPipeline,
        )
        from .cosmos import (
            Cosmos2TextToImagePipeline,
            Cosmos2VideoToWorldPipeline,
            CosmosTextToWorldPipeline,
            CosmosVideoToWorldPipeline,
        )
        from .deepfloyd_if import (
            IFImg2ImgPipeline,
            IFImg2ImgSuperResolutionPipeline,
            IFInpaintingPipeline,
            IFInpaintingSuperResolutionPipeline,
            IFPipeline,
            IFSuperResolutionPipeline,
        )
        from .deprecated import (
            AltDiffusionImg2ImgPipeline,
            AltDiffusionPipeline,
            CycleDiffusionPipeline,
            StableDiffusionInpaintPipelineLegacy,
            StableDiffusionModelEditingPipeline,
            StableDiffusionParadigmsPipeline,
            StableDiffusionPix2PixZeroPipeline,
            VersatileDiffusionDualGuidedPipeline,
            VersatileDiffusionImageVariationPipeline,
            VersatileDiffusionPipeline,
            VersatileDiffusionTextToImagePipeline,
            VQDiffusionPipeline,
        )
        from .easyanimate import (
            EasyAnimateControlPipeline,
            EasyAnimateInpaintPipeline,
            EasyAnimatePipeline,
        )
        from .flux import (
            FluxControlImg2ImgPipeline,
            FluxControlInpaintPipeline,
            FluxControlNetImg2ImgPipeline,
            FluxControlNetInpaintPipeline,
            FluxControlNetPipeline,
            FluxControlPipeline,
            FluxFillPipeline,
            FluxImg2ImgPipeline,
            FluxInpaintPipeline,
            FluxKontextInpaintPipeline,
            FluxKontextPipeline,
            FluxPipeline,
            FluxPriorReduxPipeline,
            ReduxImageEncoder,
        )
        from .hidream_image import HiDreamImagePipeline
        from .hunyuan_image import HunyuanImagePipeline, HunyuanImageRefinerPipeline
        from .hunyuan_video import (
            HunyuanSkyreelsImageToVideoPipeline,
            HunyuanVideoFramepackPipeline,
            HunyuanVideoImageToVideoPipeline,
            HunyuanVideoPipeline,
        )
        from .hunyuandit import HunyuanDiTPipeline
        from .i2vgen_xl import I2VGenXLPipeline
        from .kandinsky import (
            KandinskyCombinedPipeline,
            KandinskyImg2ImgCombinedPipeline,
            KandinskyImg2ImgPipeline,
            KandinskyInpaintCombinedPipeline,
            KandinskyInpaintPipeline,
            KandinskyPipeline,
            KandinskyPriorPipeline,
        )
        from .kandinsky2_2 import (
            KandinskyV22CombinedPipeline,
            KandinskyV22ControlnetImg2ImgPipeline,
            KandinskyV22ControlnetPipeline,
            KandinskyV22Img2ImgCombinedPipeline,
            KandinskyV22Img2ImgPipeline,
            KandinskyV22InpaintCombinedPipeline,
            KandinskyV22InpaintPipeline,
            KandinskyV22Pipeline,
            KandinskyV22PriorEmb2EmbPipeline,
            KandinskyV22PriorPipeline,
        )
        from .kandinsky3 import (
            Kandinsky3Img2ImgPipeline,
            Kandinsky3Pipeline,
        )
        from .kandinsky5 import Kandinsky5T2VPipeline
        from .latent_consistency_models import (
            LatentConsistencyModelImg2ImgPipeline,
            LatentConsistencyModelPipeline,
        )
        from .latent_diffusion import LDMTextToImagePipeline
        from .latte import LattePipeline
        from .ledits_pp import (
            LEditsPPDiffusionPipelineOutput,
            LEditsPPInversionPipelineOutput,
            LEditsPPPipelineStableDiffusion,
            LEditsPPPipelineStableDiffusionXL,
        )
        from .ltx import LTXConditionPipeline, LTXImageToVideoPipeline, LTXLatentUpsamplePipeline, LTXPipeline
        from .lucy import LucyEditPipeline
        from .lumina import LuminaPipeline, LuminaText2ImgPipeline
        from .lumina2 import Lumina2Pipeline, Lumina2Text2ImgPipeline
        from .marigold import (
            MarigoldDepthPipeline,
            MarigoldIntrinsicsPipeline,
            MarigoldNormalsPipeline,
        )
        from .mochi import MochiPipeline
        from .musicldm import MusicLDMPipeline
        from .omnigen import OmniGenPipeline
        from .pag import (
            AnimateDiffPAGPipeline,
            HunyuanDiTPAGPipeline,
            KolorsPAGPipeline,
            PixArtSigmaPAGPipeline,
            SanaPAGPipeline,
            StableDiffusion3PAGImg2ImgPipeline,
            StableDiffusion3PAGPipeline,
            StableDiffusionControlNetPAGInpaintPipeline,
            StableDiffusionControlNetPAGPipeline,
            StableDiffusionPAGImg2ImgPipeline,
            StableDiffusionPAGInpaintPipeline,
            StableDiffusionPAGPipeline,
            StableDiffusionXLControlNetPAGImg2ImgPipeline,
            StableDiffusionXLControlNetPAGPipeline,
            StableDiffusionXLPAGImg2ImgPipeline,
            StableDiffusionXLPAGInpaintPipeline,
            StableDiffusionXLPAGPipeline,
        )
        from .paint_by_example import PaintByExamplePipeline
        from .pia import PIAPipeline
        from .pixart_alpha import PixArtAlphaPipeline, PixArtSigmaPipeline
        from .prx import PRXPipeline
        from .qwenimage import (
            QwenImageControlNetInpaintPipeline,
            QwenImageControlNetPipeline,
            QwenImageEditInpaintPipeline,
            QwenImageEditPipeline,
            QwenImageEditPlusPipeline,
            QwenImageImg2ImgPipeline,
            QwenImageInpaintPipeline,
            QwenImagePipeline,
        )
        from .sana import SanaControlNetPipeline, SanaPipeline, SanaSprintImg2ImgPipeline, SanaSprintPipeline
        from .semantic_stable_diffusion import SemanticStableDiffusionPipeline
        from .shap_e import ShapEImg2ImgPipeline, ShapEPipeline
        from .stable_audio import StableAudioPipeline, StableAudioProjectionModel
        from .stable_cascade import (
            StableCascadeCombinedPipeline,
            StableCascadeDecoderPipeline,
            StableCascadePriorPipeline,
        )
        from .stable_diffusion import (
            CLIPImageProjection,
            StableDiffusionDepth2ImgPipeline,
            StableDiffusionImageVariationPipeline,
            StableDiffusionImg2ImgPipeline,
            StableDiffusionInpaintPipeline,
            StableDiffusionInstructPix2PixPipeline,
            StableDiffusionLatentUpscalePipeline,
            StableDiffusionPipeline,
            StableDiffusionUpscalePipeline,
            StableUnCLIPImg2ImgPipeline,
            StableUnCLIPPipeline,
        )
        from .stable_diffusion_3 import (
            StableDiffusion3Img2ImgPipeline,
            StableDiffusion3InpaintPipeline,
            StableDiffusion3Pipeline,
        )
        from .stable_diffusion_attend_and_excite import StableDiffusionAttendAndExcitePipeline
        from .stable_diffusion_diffedit import StableDiffusionDiffEditPipeline
        from .stable_diffusion_gligen import StableDiffusionGLIGENPipeline, StableDiffusionGLIGENTextImagePipeline
        from .stable_diffusion_ldm3d import StableDiffusionLDM3DPipeline
        from .stable_diffusion_panorama import StableDiffusionPanoramaPipeline
        from .stable_diffusion_safe import StableDiffusionPipelineSafe
        from .stable_diffusion_sag import StableDiffusionSAGPipeline
        from .stable_diffusion_xl import (
            StableDiffusionXLImg2ImgPipeline,
            StableDiffusionXLInpaintPipeline,
            StableDiffusionXLInstructPix2PixPipeline,
            StableDiffusionXLPipeline,
        )
        from .stable_video_diffusion import StableVideoDiffusionPipeline
        from .t2i_adapter import (
            StableDiffusionAdapterPipeline,
            StableDiffusionXLAdapterPipeline,
        )
        from .text_to_video_synthesis import (
            TextToVideoSDPipeline,
            TextToVideoZeroPipeline,
            TextToVideoZeroSDXLPipeline,
            VideoToVideoSDPipeline,
        )
        from .unclip import UnCLIPImageVariationPipeline, UnCLIPPipeline
        from .unidiffuser import (
            ImageTextPipelineOutput,
            UniDiffuserModel,
            UniDiffuserPipeline,
            UniDiffuserTextDecoder,
        )
        from .visualcloze import VisualClozeGenerationPipeline, VisualClozePipeline
        from .wan import WanImageToVideoPipeline, WanPipeline, WanVACEPipeline, WanVideoToVideoPipeline
        from .wuerstchen import (
            WuerstchenCombinedPipeline,
            WuerstchenDecoderPipeline,
            WuerstchenPriorPipeline,
        )

        try:
            if not is_onnx_available():
                raise OptionalDependencyNotAvailable()
        except OptionalDependencyNotAvailable:
            from ..utils.dummy_onnx_objects import *  # noqa F403

        else:
            from .onnx_utils import OnnxRuntimeModel

        try:
            if not (is_torch_available() and is_transformers_available() and is_onnx_available()):
                raise OptionalDependencyNotAvailable()
        except OptionalDependencyNotAvailable:
            from ..utils.dummy_torch_and_transformers_and_onnx_objects import *
        else:
            from .stable_diffusion import (
                OnnxStableDiffusionImg2ImgPipeline,
                OnnxStableDiffusionInpaintPipeline,
                OnnxStableDiffusionPipeline,
                OnnxStableDiffusionUpscalePipeline,
                StableDiffusionOnnxPipeline,
            )

        try:
            if not (is_torch_available() and is_transformers_available() and is_k_diffusion_available()):
                raise OptionalDependencyNotAvailable()
        except OptionalDependencyNotAvailable:
            from ..utils.dummy_torch_and_transformers_and_k_diffusion_objects import *
        else:
            from .stable_diffusion_k_diffusion import (
                StableDiffusionKDiffusionPipeline,
                StableDiffusionXLKDiffusionPipeline,
            )

        try:
            if not (is_torch_available() and is_transformers_available() and is_sentencepiece_available()):
                raise OptionalDependencyNotAvailable()
        except OptionalDependencyNotAvailable:
            from ..utils.dummy_torch_and_transformers_and_sentencepiece_objects import *
        else:
            from .kolors import (
                KolorsImg2ImgPipeline,
                KolorsPipeline,
            )

        try:
            if not (is_torch_available() and is_transformers_available() and is_opencv_available()):
                raise OptionalDependencyNotAvailable()
        except OptionalDependencyNotAvailable:
            from ..utils.dummy_torch_and_transformers_and_opencv_objects import *
        else:
            from .consisid import ConsisIDPipeline

        try:
            if not is_flax_available():
                raise OptionalDependencyNotAvailable()
        except OptionalDependencyNotAvailable:
            from ..utils.dummy_flax_objects import *  # noqa F403
        else:
            from .pipeline_flax_utils import FlaxDiffusionPipeline

        try:
            if not (is_flax_available() and is_transformers_available()):
                raise OptionalDependencyNotAvailable()
        except OptionalDependencyNotAvailable:
            from ..utils.dummy_flax_and_transformers_objects import *
        else:
            from .controlnet import FlaxStableDiffusionControlNetPipeline
            from .stable_diffusion import (
                FlaxStableDiffusionImg2ImgPipeline,
                FlaxStableDiffusionInpaintPipeline,
                FlaxStableDiffusionPipeline,
            )
            from .stable_diffusion_xl import (
                FlaxStableDiffusionXLPipeline,
            )

        try:
            if not (is_transformers_available() and is_torch_available() and is_note_seq_available()):
                raise OptionalDependencyNotAvailable()
        except OptionalDependencyNotAvailable:
            from ..utils.dummy_transformers_and_torch_and_note_seq_objects import *  # noqa F403

        else:
            from .deprecated import (
                MidiProcessor,
                SpectrogramDiffusionPipeline,
            )

        from .skyreels_v2 import (
            SkyReelsV2DiffusionForcingImageToVideoPipeline,
            SkyReelsV2DiffusionForcingPipeline,
            SkyReelsV2DiffusionForcingVideoToVideoPipeline,
            SkyReelsV2ImageToVideoPipeline,
            SkyReelsV2Pipeline,
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
