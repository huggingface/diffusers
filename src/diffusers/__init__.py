__version__ = "0.36.0.dev0"

from typing import TYPE_CHECKING

from .utils import (
    DIFFUSERS_SLOW_IMPORT,
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_accelerate_available,
    is_bitsandbytes_available,
    is_flax_available,
    is_gguf_available,
    is_k_diffusion_available,
    is_librosa_available,
    is_note_seq_available,
    is_nvidia_modelopt_available,
    is_onnx_available,
    is_opencv_available,
    is_optimum_quanto_available,
    is_scipy_available,
    is_sentencepiece_available,
    is_torch_available,
    is_torchao_available,
    is_torchsde_available,
    is_transformers_available,
)


# Lazy Import based on
# https://github.com/huggingface/transformers/blob/main/src/transformers/__init__.py

# When adding a new object to this init, please add it to `_import_structure`. The `_import_structure` is a dictionary submodule to list of object names,
# and is used to defer the actual importing for when the objects are requested.
# This way `import diffusers` provides the names in the namespace without actually importing anything (and especially none of the backends).

_import_structure = {
    "configuration_utils": ["ConfigMixin"],
    "guiders": [],
    "hooks": [],
    "loaders": ["FromOriginalModelMixin"],
    "models": [],
    "modular_pipelines": [],
    "pipelines": [],
    "quantizers.pipe_quant_config": ["PipelineQuantizationConfig"],
    "quantizers.quantization_config": [],
    "schedulers": [],
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
}

try:
    if not is_torch_available() and not is_accelerate_available() and not is_bitsandbytes_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils import dummy_bitsandbytes_objects

    _import_structure["utils.dummy_bitsandbytes_objects"] = [
        name for name in dir(dummy_bitsandbytes_objects) if not name.startswith("_")
    ]
else:
    _import_structure["quantizers.quantization_config"].append("BitsAndBytesConfig")

try:
    if not is_torch_available() and not is_accelerate_available() and not is_gguf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils import dummy_gguf_objects

    _import_structure["utils.dummy_gguf_objects"] = [
        name for name in dir(dummy_gguf_objects) if not name.startswith("_")
    ]
else:
    _import_structure["quantizers.quantization_config"].append("GGUFQuantizationConfig")

try:
    if not is_torch_available() and not is_accelerate_available() and not is_torchao_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils import dummy_torchao_objects

    _import_structure["utils.dummy_torchao_objects"] = [
        name for name in dir(dummy_torchao_objects) if not name.startswith("_")
    ]
else:
    _import_structure["quantizers.quantization_config"].append("TorchAoConfig")

try:
    if not is_torch_available() and not is_accelerate_available() and not is_optimum_quanto_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils import dummy_optimum_quanto_objects

    _import_structure["utils.dummy_optimum_quanto_objects"] = [
        name for name in dir(dummy_optimum_quanto_objects) if not name.startswith("_")
    ]
else:
    _import_structure["quantizers.quantization_config"].append("QuantoConfig")

try:
    if not is_torch_available() and not is_accelerate_available() and not is_nvidia_modelopt_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils import dummy_nvidia_modelopt_objects

    _import_structure["utils.dummy_nvidia_modelopt_objects"] = [
        name for name in dir(dummy_nvidia_modelopt_objects) if not name.startswith("_")
    ]
else:
    _import_structure["quantizers.quantization_config"].append("NVIDIAModelOptConfig")

try:
    if not is_onnx_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils import dummy_onnx_objects  # noqa F403

    _import_structure["utils.dummy_onnx_objects"] = [
        name for name in dir(dummy_onnx_objects) if not name.startswith("_")
    ]

else:
    _import_structure["pipelines"].extend(["OnnxRuntimeModel"])

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils import dummy_pt_objects  # noqa F403

    _import_structure["utils.dummy_pt_objects"] = [name for name in dir(dummy_pt_objects) if not name.startswith("_")]

else:
    _import_structure["guiders"].extend(
        [
            "AdaptiveProjectedGuidance",
            "AutoGuidance",
            "ClassifierFreeGuidance",
            "ClassifierFreeZeroStarGuidance",
            "FrequencyDecoupledGuidance",
            "PerturbedAttentionGuidance",
            "SkipLayerGuidance",
            "SmoothedEnergyGuidance",
            "TangentialClassifierFreeGuidance",
        ]
    )
    _import_structure["hooks"].extend(
        [
            "FasterCacheConfig",
            "FirstBlockCacheConfig",
            "HookRegistry",
            "LayerSkipConfig",
            "PyramidAttentionBroadcastConfig",
            "SmoothedEnergyGuidanceConfig",
            "apply_faster_cache",
            "apply_first_block_cache",
            "apply_layer_skip",
            "apply_pyramid_attention_broadcast",
        ]
    )
    _import_structure["models"].extend(
        [
            "AllegroTransformer3DModel",
            "AsymmetricAutoencoderKL",
            "AttentionBackendName",
            "AuraFlowTransformer2DModel",
            "AutoencoderDC",
            "AutoencoderKL",
            "AutoencoderKLAllegro",
            "AutoencoderKLCogVideoX",
            "AutoencoderKLCosmos",
            "AutoencoderKLHunyuanVideo",
            "AutoencoderKLLTXVideo",
            "AutoencoderKLMagvit",
            "AutoencoderKLMochi",
            "AutoencoderKLQwenImage",
            "AutoencoderKLTemporalDecoder",
            "AutoencoderKLWan",
            "AutoencoderOobleck",
            "AutoencoderTiny",
            "AutoModel",
            "BriaTransformer2DModel",
            "CacheMixin",
            "ChromaTransformer2DModel",
            "CogVideoXTransformer3DModel",
            "CogView3PlusTransformer2DModel",
            "CogView4Transformer2DModel",
            "ConsisIDTransformer3DModel",
            "ConsistencyDecoderVAE",
            "ContextParallelConfig",
            "ControlNetModel",
            "ControlNetUnionModel",
            "ControlNetXSAdapter",
            "CosmosTransformer3DModel",
            "DiTTransformer2DModel",
            "EasyAnimateTransformer3DModel",
            "FluxControlNetModel",
            "FluxMultiControlNetModel",
            "FluxTransformer2DModel",
            "HiDreamImageTransformer2DModel",
            "HunyuanDiT2DControlNetModel",
            "HunyuanDiT2DModel",
            "HunyuanDiT2DMultiControlNetModel",
            "HunyuanVideoFramepackTransformer3DModel",
            "HunyuanVideoTransformer3DModel",
            "I2VGenXLUNet",
            "Kandinsky3UNet",
            "LatteTransformer3DModel",
            "LTXVideoTransformer3DModel",
            "Lumina2Transformer2DModel",
            "LuminaNextDiT2DModel",
            "MochiTransformer3DModel",
            "ModelMixin",
            "MotionAdapter",
            "MultiAdapter",
            "MultiControlNetModel",
            "OmniGenTransformer2DModel",
            "ParallelConfig",
            "PixArtTransformer2DModel",
            "PriorTransformer",
            "QwenImageControlNetModel",
            "QwenImageMultiControlNetModel",
            "QwenImageTransformer2DModel",
            "SanaControlNetModel",
            "SanaTransformer2DModel",
            "SD3ControlNetModel",
            "SD3MultiControlNetModel",
            "SD3Transformer2DModel",
            "SkyReelsV2Transformer3DModel",
            "SparseControlNetModel",
            "StableAudioDiTModel",
            "StableCascadeUNet",
            "T2IAdapter",
            "T5FilmDecoder",
            "Transformer2DModel",
            "TransformerTemporalModel",
            "UNet1DModel",
            "UNet2DConditionModel",
            "UNet2DModel",
            "UNet3DConditionModel",
            "UNetControlNetXSModel",
            "UNetMotionModel",
            "UNetSpatioTemporalConditionModel",
            "UVit2DModel",
            "VQModel",
            "WanTransformer3DModel",
            "WanVACETransformer3DModel",
            "attention_backend",
        ]
    )
    _import_structure["modular_pipelines"].extend(
        [
            "ComponentsManager",
            "ComponentSpec",
            "ModularPipeline",
            "ModularPipelineBlocks",
        ]
    )
    _import_structure["optimization"] = [
        "get_constant_schedule",
        "get_constant_schedule_with_warmup",
        "get_cosine_schedule_with_warmup",
        "get_cosine_with_hard_restarts_schedule_with_warmup",
        "get_linear_schedule_with_warmup",
        "get_polynomial_decay_schedule_with_warmup",
        "get_scheduler",
    ]
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
            "StableDiffusionMixin",
        ]
    )
    _import_structure["quantizers"] = ["DiffusersQuantizer"]
    _import_structure["schedulers"].extend(
        [
            "AmusedScheduler",
            "CMStochasticIterativeScheduler",
            "CogVideoXDDIMScheduler",
            "CogVideoXDPMScheduler",
            "DDIMInverseScheduler",
            "DDIMParallelScheduler",
            "DDIMScheduler",
            "DDPMParallelScheduler",
            "DDPMScheduler",
            "DDPMWuerstchenScheduler",
            "DEISMultistepScheduler",
            "DPMSolverMultistepInverseScheduler",
            "DPMSolverMultistepScheduler",
            "DPMSolverSinglestepScheduler",
            "EDMDPMSolverMultistepScheduler",
            "EDMEulerScheduler",
            "EulerAncestralDiscreteScheduler",
            "EulerDiscreteScheduler",
            "FlowMatchEulerDiscreteScheduler",
            "FlowMatchHeunDiscreteScheduler",
            "FlowMatchLCMScheduler",
            "HeunDiscreteScheduler",
            "IPNDMScheduler",
            "KarrasVeScheduler",
            "KDPM2AncestralDiscreteScheduler",
            "KDPM2DiscreteScheduler",
            "LCMScheduler",
            "PNDMScheduler",
            "RePaintScheduler",
            "SASolverScheduler",
            "SchedulerMixin",
            "SCMScheduler",
            "ScoreSdeVeScheduler",
            "TCDScheduler",
            "UnCLIPScheduler",
            "UniPCMultistepScheduler",
            "VQDiffusionScheduler",
        ]
    )
    _import_structure["training_utils"] = ["EMAModel"]

try:
    if not (is_torch_available() and is_scipy_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils import dummy_torch_and_scipy_objects  # noqa F403

    _import_structure["utils.dummy_torch_and_scipy_objects"] = [
        name for name in dir(dummy_torch_and_scipy_objects) if not name.startswith("_")
    ]

else:
    _import_structure["schedulers"].extend(["LMSDiscreteScheduler"])

try:
    if not (is_torch_available() and is_torchsde_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils import dummy_torch_and_torchsde_objects  # noqa F403

    _import_structure["utils.dummy_torch_and_torchsde_objects"] = [
        name for name in dir(dummy_torch_and_torchsde_objects) if not name.startswith("_")
    ]

else:
    _import_structure["schedulers"].extend(["CosineDPMSolverMultistepScheduler", "DPMSolverSDEScheduler"])

try:
    if not (is_torch_available() and is_transformers_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils import dummy_torch_and_transformers_objects  # noqa F403

    _import_structure["utils.dummy_torch_and_transformers_objects"] = [
        name for name in dir(dummy_torch_and_transformers_objects) if not name.startswith("_")
    ]

else:
    _import_structure["modular_pipelines"].extend(
        [
            "FluxAutoBlocks",
            "FluxModularPipeline",
            "QwenImageAutoBlocks",
            "QwenImageEditAutoBlocks",
            "QwenImageEditModularPipeline",
            "QwenImageEditPlusAutoBlocks",
            "QwenImageEditPlusModularPipeline",
            "QwenImageModularPipeline",
            "StableDiffusionXLAutoBlocks",
            "StableDiffusionXLModularPipeline",
            "WanAutoBlocks",
            "WanModularPipeline",
        ]
    )
    _import_structure["pipelines"].extend(
        [
            "AllegroPipeline",
            "AltDiffusionImg2ImgPipeline",
            "AltDiffusionPipeline",
            "AmusedImg2ImgPipeline",
            "AmusedInpaintPipeline",
            "AmusedPipeline",
            "AnimateDiffControlNetPipeline",
            "AnimateDiffPAGPipeline",
            "AnimateDiffPipeline",
            "AnimateDiffSDXLPipeline",
            "AnimateDiffSparseControlNetPipeline",
            "AnimateDiffVideoToVideoControlNetPipeline",
            "AnimateDiffVideoToVideoPipeline",
            "AudioLDM2Pipeline",
            "AudioLDM2ProjectionModel",
            "AudioLDM2UNet2DConditionModel",
            "AudioLDMPipeline",
            "AuraFlowPipeline",
            "BlipDiffusionControlNetPipeline",
            "BlipDiffusionPipeline",
            "BriaPipeline",
            "ChromaImg2ImgPipeline",
            "ChromaPipeline",
            "CLIPImageProjection",
            "CogVideoXFunControlPipeline",
            "CogVideoXImageToVideoPipeline",
            "CogVideoXPipeline",
            "CogVideoXVideoToVideoPipeline",
            "CogView3PlusPipeline",
            "CogView4ControlPipeline",
            "CogView4Pipeline",
            "ConsisIDPipeline",
            "Cosmos2TextToImagePipeline",
            "Cosmos2VideoToWorldPipeline",
            "CosmosTextToWorldPipeline",
            "CosmosVideoToWorldPipeline",
            "CycleDiffusionPipeline",
            "EasyAnimateControlPipeline",
            "EasyAnimateInpaintPipeline",
            "EasyAnimatePipeline",
            "FluxControlImg2ImgPipeline",
            "FluxControlInpaintPipeline",
            "FluxControlNetImg2ImgPipeline",
            "FluxControlNetInpaintPipeline",
            "FluxControlNetPipeline",
            "FluxControlPipeline",
            "FluxFillPipeline",
            "FluxImg2ImgPipeline",
            "FluxInpaintPipeline",
            "FluxKontextInpaintPipeline",
            "FluxKontextPipeline",
            "FluxPipeline",
            "FluxPriorReduxPipeline",
            "HiDreamImagePipeline",
            "HunyuanDiTControlNetPipeline",
            "HunyuanDiTPAGPipeline",
            "HunyuanDiTPipeline",
            "HunyuanSkyreelsImageToVideoPipeline",
            "HunyuanVideoFramepackPipeline",
            "HunyuanVideoImageToVideoPipeline",
            "HunyuanVideoPipeline",
            "I2VGenXLPipeline",
            "IFImg2ImgPipeline",
            "IFImg2ImgSuperResolutionPipeline",
            "IFInpaintingPipeline",
            "IFInpaintingSuperResolutionPipeline",
            "IFPipeline",
            "IFSuperResolutionPipeline",
            "ImageTextPipelineOutput",
            "Kandinsky3Img2ImgPipeline",
            "Kandinsky3Pipeline",
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
            "LatentConsistencyModelImg2ImgPipeline",
            "LatentConsistencyModelPipeline",
            "LattePipeline",
            "LDMTextToImagePipeline",
            "LEditsPPPipelineStableDiffusion",
            "LEditsPPPipelineStableDiffusionXL",
            "LTXConditionPipeline",
            "LTXImageToVideoPipeline",
            "LTXLatentUpsamplePipeline",
            "LTXPipeline",
            "LucyEditPipeline",
            "Lumina2Pipeline",
            "Lumina2Text2ImgPipeline",
            "LuminaPipeline",
            "LuminaText2ImgPipeline",
            "MarigoldDepthPipeline",
            "MarigoldIntrinsicsPipeline",
            "MarigoldNormalsPipeline",
            "MochiPipeline",
            "MusicLDMPipeline",
            "OmniGenPipeline",
            "PaintByExamplePipeline",
            "PIAPipeline",
            "PixArtAlphaPipeline",
            "PixArtSigmaPAGPipeline",
            "PixArtSigmaPipeline",
            "QwenImageControlNetInpaintPipeline",
            "QwenImageControlNetPipeline",
            "QwenImageEditInpaintPipeline",
            "QwenImageEditPipeline",
            "QwenImageEditPlusPipeline",
            "QwenImageImg2ImgPipeline",
            "QwenImageInpaintPipeline",
            "QwenImagePipeline",
            "ReduxImageEncoder",
            "SanaControlNetPipeline",
            "SanaPAGPipeline",
            "SanaPipeline",
            "SanaSprintImg2ImgPipeline",
            "SanaSprintPipeline",
            "SemanticStableDiffusionPipeline",
            "ShapEImg2ImgPipeline",
            "ShapEPipeline",
            "SkyReelsV2DiffusionForcingImageToVideoPipeline",
            "SkyReelsV2DiffusionForcingPipeline",
            "SkyReelsV2DiffusionForcingVideoToVideoPipeline",
            "SkyReelsV2ImageToVideoPipeline",
            "SkyReelsV2Pipeline",
            "StableAudioPipeline",
            "StableAudioProjectionModel",
            "StableCascadeCombinedPipeline",
            "StableCascadeDecoderPipeline",
            "StableCascadePriorPipeline",
            "StableDiffusion3ControlNetInpaintingPipeline",
            "StableDiffusion3ControlNetPipeline",
            "StableDiffusion3Img2ImgPipeline",
            "StableDiffusion3InpaintPipeline",
            "StableDiffusion3PAGImg2ImgPipeline",
            "StableDiffusion3PAGImg2ImgPipeline",
            "StableDiffusion3PAGPipeline",
            "StableDiffusion3Pipeline",
            "StableDiffusionAdapterPipeline",
            "StableDiffusionAttendAndExcitePipeline",
            "StableDiffusionControlNetImg2ImgPipeline",
            "StableDiffusionControlNetInpaintPipeline",
            "StableDiffusionControlNetPAGInpaintPipeline",
            "StableDiffusionControlNetPAGPipeline",
            "StableDiffusionControlNetPipeline",
            "StableDiffusionControlNetXSPipeline",
            "StableDiffusionDepth2ImgPipeline",
            "StableDiffusionDiffEditPipeline",
            "StableDiffusionGLIGENPipeline",
            "StableDiffusionGLIGENTextImagePipeline",
            "StableDiffusionImageVariationPipeline",
            "StableDiffusionImg2ImgPipeline",
            "StableDiffusionInpaintPipeline",
            "StableDiffusionInpaintPipelineLegacy",
            "StableDiffusionInstructPix2PixPipeline",
            "StableDiffusionLatentUpscalePipeline",
            "StableDiffusionLDM3DPipeline",
            "StableDiffusionModelEditingPipeline",
            "StableDiffusionPAGImg2ImgPipeline",
            "StableDiffusionPAGInpaintPipeline",
            "StableDiffusionPAGPipeline",
            "StableDiffusionPanoramaPipeline",
            "StableDiffusionParadigmsPipeline",
            "StableDiffusionPipeline",
            "StableDiffusionPipelineSafe",
            "StableDiffusionPix2PixZeroPipeline",
            "StableDiffusionSAGPipeline",
            "StableDiffusionUpscalePipeline",
            "StableDiffusionXLAdapterPipeline",
            "StableDiffusionXLControlNetImg2ImgPipeline",
            "StableDiffusionXLControlNetInpaintPipeline",
            "StableDiffusionXLControlNetPAGImg2ImgPipeline",
            "StableDiffusionXLControlNetPAGPipeline",
            "StableDiffusionXLControlNetPipeline",
            "StableDiffusionXLControlNetUnionImg2ImgPipeline",
            "StableDiffusionXLControlNetUnionInpaintPipeline",
            "StableDiffusionXLControlNetUnionPipeline",
            "StableDiffusionXLControlNetXSPipeline",
            "StableDiffusionXLImg2ImgPipeline",
            "StableDiffusionXLInpaintPipeline",
            "StableDiffusionXLInstructPix2PixPipeline",
            "StableDiffusionXLPAGImg2ImgPipeline",
            "StableDiffusionXLPAGInpaintPipeline",
            "StableDiffusionXLPAGPipeline",
            "StableDiffusionXLPipeline",
            "StableUnCLIPImg2ImgPipeline",
            "StableUnCLIPPipeline",
            "StableVideoDiffusionPipeline",
            "TextToVideoSDPipeline",
            "TextToVideoZeroPipeline",
            "TextToVideoZeroSDXLPipeline",
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
            "VisualClozeGenerationPipeline",
            "VisualClozePipeline",
            "VQDiffusionPipeline",
            "WanImageToVideoPipeline",
            "WanPipeline",
            "WanVACEPipeline",
            "WanVideoToVideoPipeline",
            "WuerstchenCombinedPipeline",
            "WuerstchenDecoderPipeline",
            "WuerstchenPriorPipeline",
        ]
    )


try:
    if not (is_torch_available() and is_transformers_available() and is_opencv_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils import dummy_torch_and_transformers_and_opencv_objects  # noqa F403

    _import_structure["utils.dummy_torch_and_transformers_and_opencv_objects"] = [
        name for name in dir(dummy_torch_and_transformers_and_opencv_objects) if not name.startswith("_")
    ]

else:
    _import_structure["pipelines"].extend(["ConsisIDPipeline"])

try:
    if not (is_torch_available() and is_transformers_available() and is_k_diffusion_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils import dummy_torch_and_transformers_and_k_diffusion_objects  # noqa F403

    _import_structure["utils.dummy_torch_and_transformers_and_k_diffusion_objects"] = [
        name for name in dir(dummy_torch_and_transformers_and_k_diffusion_objects) if not name.startswith("_")
    ]

else:
    _import_structure["pipelines"].extend(["StableDiffusionKDiffusionPipeline", "StableDiffusionXLKDiffusionPipeline"])

try:
    if not (is_torch_available() and is_transformers_available() and is_sentencepiece_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils import dummy_torch_and_transformers_and_sentencepiece_objects  # noqa F403

    _import_structure["utils.dummy_torch_and_transformers_and_sentencepiece_objects"] = [
        name for name in dir(dummy_torch_and_transformers_and_sentencepiece_objects) if not name.startswith("_")
    ]

else:
    _import_structure["pipelines"].extend(["KolorsImg2ImgPipeline", "KolorsPAGPipeline", "KolorsPipeline"])

try:
    if not (is_torch_available() and is_transformers_available() and is_onnx_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils import dummy_torch_and_transformers_and_onnx_objects  # noqa F403

    _import_structure["utils.dummy_torch_and_transformers_and_onnx_objects"] = [
        name for name in dir(dummy_torch_and_transformers_and_onnx_objects) if not name.startswith("_")
    ]

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
    from .utils import dummy_torch_and_librosa_objects  # noqa F403

    _import_structure["utils.dummy_torch_and_librosa_objects"] = [
        name for name in dir(dummy_torch_and_librosa_objects) if not name.startswith("_")
    ]

else:
    _import_structure["pipelines"].extend(["AudioDiffusionPipeline", "Mel"])

try:
    if not (is_transformers_available() and is_torch_available() and is_note_seq_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils import dummy_transformers_and_torch_and_note_seq_objects  # noqa F403

    _import_structure["utils.dummy_transformers_and_torch_and_note_seq_objects"] = [
        name for name in dir(dummy_transformers_and_torch_and_note_seq_objects) if not name.startswith("_")
    ]


else:
    _import_structure["pipelines"].extend(["SpectrogramDiffusionPipeline"])

try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils import dummy_flax_objects  # noqa F403

    _import_structure["utils.dummy_flax_objects"] = [
        name for name in dir(dummy_flax_objects) if not name.startswith("_")
    ]


else:
    _import_structure["models.controlnets.controlnet_flax"] = ["FlaxControlNetModel"]
    _import_structure["models.modeling_flax_utils"] = ["FlaxModelMixin"]
    _import_structure["models.unets.unet_2d_condition_flax"] = ["FlaxUNet2DConditionModel"]
    _import_structure["models.vae_flax"] = ["FlaxAutoencoderKL"]
    _import_structure["pipelines"].extend(["FlaxDiffusionPipeline"])
    _import_structure["schedulers"].extend(
        [
            "FlaxDDIMScheduler",
            "FlaxDDPMScheduler",
            "FlaxDPMSolverMultistepScheduler",
            "FlaxEulerDiscreteScheduler",
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
    from .utils import dummy_flax_and_transformers_objects  # noqa F403

    _import_structure["utils.dummy_flax_and_transformers_objects"] = [
        name for name in dir(dummy_flax_and_transformers_objects) if not name.startswith("_")
    ]


else:
    _import_structure["pipelines"].extend(
        [
            "FlaxStableDiffusionControlNetPipeline",
            "FlaxStableDiffusionImg2ImgPipeline",
            "FlaxStableDiffusionInpaintPipeline",
            "FlaxStableDiffusionPipeline",
            "FlaxStableDiffusionXLPipeline",
        ]
    )

try:
    if not (is_note_seq_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils import dummy_note_seq_objects  # noqa F403

    _import_structure["utils.dummy_note_seq_objects"] = [
        name for name in dir(dummy_note_seq_objects) if not name.startswith("_")
    ]


else:
    _import_structure["pipelines"].extend(["MidiProcessor"])

if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    from .configuration_utils import ConfigMixin
    from .quantizers import PipelineQuantizationConfig

    try:
        if not is_bitsandbytes_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from .utils.dummy_bitsandbytes_objects import *
    else:
        from .quantizers.quantization_config import BitsAndBytesConfig

    try:
        if not is_gguf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from .utils.dummy_gguf_objects import *
    else:
        from .quantizers.quantization_config import GGUFQuantizationConfig

    try:
        if not is_torchao_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from .utils.dummy_torchao_objects import *
    else:
        from .quantizers.quantization_config import TorchAoConfig

    try:
        if not is_optimum_quanto_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from .utils.dummy_optimum_quanto_objects import *
    else:
        from .quantizers.quantization_config import QuantoConfig

    try:
        if not is_nvidia_modelopt_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from .utils.dummy_nvidia_modelopt_objects import *
    else:
        from .quantizers.quantization_config import NVIDIAModelOptConfig

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
        from .guiders import (
            AdaptiveProjectedGuidance,
            AutoGuidance,
            ClassifierFreeGuidance,
            ClassifierFreeZeroStarGuidance,
            FrequencyDecoupledGuidance,
            PerturbedAttentionGuidance,
            SkipLayerGuidance,
            SmoothedEnergyGuidance,
            TangentialClassifierFreeGuidance,
        )
        from .hooks import (
            FasterCacheConfig,
            FirstBlockCacheConfig,
            HookRegistry,
            LayerSkipConfig,
            PyramidAttentionBroadcastConfig,
            SmoothedEnergyGuidanceConfig,
            apply_faster_cache,
            apply_first_block_cache,
            apply_layer_skip,
            apply_pyramid_attention_broadcast,
        )
        from .models import (
            AllegroTransformer3DModel,
            AsymmetricAutoencoderKL,
            AttentionBackendName,
            AuraFlowTransformer2DModel,
            AutoencoderDC,
            AutoencoderKL,
            AutoencoderKLAllegro,
            AutoencoderKLCogVideoX,
            AutoencoderKLCosmos,
            AutoencoderKLHunyuanVideo,
            AutoencoderKLLTXVideo,
            AutoencoderKLMagvit,
            AutoencoderKLMochi,
            AutoencoderKLQwenImage,
            AutoencoderKLTemporalDecoder,
            AutoencoderKLWan,
            AutoencoderOobleck,
            AutoencoderTiny,
            AutoModel,
            BriaTransformer2DModel,
            CacheMixin,
            ChromaTransformer2DModel,
            CogVideoXTransformer3DModel,
            CogView3PlusTransformer2DModel,
            CogView4Transformer2DModel,
            ConsisIDTransformer3DModel,
            ConsistencyDecoderVAE,
            ContextParallelConfig,
            ControlNetModel,
            ControlNetUnionModel,
            ControlNetXSAdapter,
            CosmosTransformer3DModel,
            DiTTransformer2DModel,
            EasyAnimateTransformer3DModel,
            FluxControlNetModel,
            FluxMultiControlNetModel,
            FluxTransformer2DModel,
            HiDreamImageTransformer2DModel,
            HunyuanDiT2DControlNetModel,
            HunyuanDiT2DModel,
            HunyuanDiT2DMultiControlNetModel,
            HunyuanVideoFramepackTransformer3DModel,
            HunyuanVideoTransformer3DModel,
            I2VGenXLUNet,
            Kandinsky3UNet,
            LatteTransformer3DModel,
            LTXVideoTransformer3DModel,
            Lumina2Transformer2DModel,
            LuminaNextDiT2DModel,
            MochiTransformer3DModel,
            ModelMixin,
            MotionAdapter,
            MultiAdapter,
            MultiControlNetModel,
            OmniGenTransformer2DModel,
            ParallelConfig,
            PixArtTransformer2DModel,
            PriorTransformer,
            QwenImageControlNetModel,
            QwenImageMultiControlNetModel,
            QwenImageTransformer2DModel,
            SanaControlNetModel,
            SanaTransformer2DModel,
            SD3ControlNetModel,
            SD3MultiControlNetModel,
            SD3Transformer2DModel,
            SkyReelsV2Transformer3DModel,
            SparseControlNetModel,
            StableAudioDiTModel,
            T2IAdapter,
            T5FilmDecoder,
            Transformer2DModel,
            TransformerTemporalModel,
            UNet1DModel,
            UNet2DConditionModel,
            UNet2DModel,
            UNet3DConditionModel,
            UNetControlNetXSModel,
            UNetMotionModel,
            UNetSpatioTemporalConditionModel,
            UVit2DModel,
            VQModel,
            WanTransformer3DModel,
            WanVACETransformer3DModel,
            attention_backend,
        )
        from .modular_pipelines import ComponentsManager, ComponentSpec, ModularPipeline, ModularPipelineBlocks
        from .optimization import (
            get_constant_schedule,
            get_constant_schedule_with_warmup,
            get_cosine_schedule_with_warmup,
            get_cosine_with_hard_restarts_schedule_with_warmup,
            get_linear_schedule_with_warmup,
            get_polynomial_decay_schedule_with_warmup,
            get_scheduler,
        )
        from .pipelines import (
            AudioPipelineOutput,
            AutoPipelineForImage2Image,
            AutoPipelineForInpainting,
            AutoPipelineForText2Image,
            BlipDiffusionControlNetPipeline,
            BlipDiffusionPipeline,
            CLIPImageProjection,
            ConsistencyModelPipeline,
            DanceDiffusionPipeline,
            DDIMPipeline,
            DDPMPipeline,
            DiffusionPipeline,
            DiTPipeline,
            ImagePipelineOutput,
            KarrasVePipeline,
            LDMPipeline,
            LDMSuperResolutionPipeline,
            PNDMPipeline,
            RePaintPipeline,
            ScoreSdeVePipeline,
            StableDiffusionMixin,
        )
        from .quantizers import DiffusersQuantizer
        from .schedulers import (
            AmusedScheduler,
            CMStochasticIterativeScheduler,
            CogVideoXDDIMScheduler,
            CogVideoXDPMScheduler,
            DDIMInverseScheduler,
            DDIMParallelScheduler,
            DDIMScheduler,
            DDPMParallelScheduler,
            DDPMScheduler,
            DDPMWuerstchenScheduler,
            DEISMultistepScheduler,
            DPMSolverMultistepInverseScheduler,
            DPMSolverMultistepScheduler,
            DPMSolverSinglestepScheduler,
            EDMDPMSolverMultistepScheduler,
            EDMEulerScheduler,
            EulerAncestralDiscreteScheduler,
            EulerDiscreteScheduler,
            FlowMatchEulerDiscreteScheduler,
            FlowMatchHeunDiscreteScheduler,
            FlowMatchLCMScheduler,
            HeunDiscreteScheduler,
            IPNDMScheduler,
            KarrasVeScheduler,
            KDPM2AncestralDiscreteScheduler,
            KDPM2DiscreteScheduler,
            LCMScheduler,
            PNDMScheduler,
            RePaintScheduler,
            SASolverScheduler,
            SchedulerMixin,
            SCMScheduler,
            ScoreSdeVeScheduler,
            TCDScheduler,
            UnCLIPScheduler,
            UniPCMultistepScheduler,
            VQDiffusionScheduler,
        )
        from .training_utils import EMAModel

    try:
        if not (is_torch_available() and is_scipy_available()):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from .utils.dummy_torch_and_scipy_objects import *  # noqa F403
    else:
        from .schedulers import LMSDiscreteScheduler

    try:
        if not (is_torch_available() and is_torchsde_available()):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from .utils.dummy_torch_and_torchsde_objects import *  # noqa F403
    else:
        from .schedulers import CosineDPMSolverMultistepScheduler, DPMSolverSDEScheduler

    try:
        if not (is_torch_available() and is_transformers_available()):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from .utils.dummy_torch_and_transformers_objects import *  # noqa F403
    else:
        from .modular_pipelines import (
            FluxAutoBlocks,
            FluxModularPipeline,
            QwenImageAutoBlocks,
            QwenImageEditAutoBlocks,
            QwenImageEditModularPipeline,
            QwenImageEditPlusAutoBlocks,
            QwenImageEditPlusModularPipeline,
            QwenImageModularPipeline,
            StableDiffusionXLAutoBlocks,
            StableDiffusionXLModularPipeline,
            WanAutoBlocks,
            WanModularPipeline,
        )
        from .pipelines import (
            AllegroPipeline,
            AltDiffusionImg2ImgPipeline,
            AltDiffusionPipeline,
            AmusedImg2ImgPipeline,
            AmusedInpaintPipeline,
            AmusedPipeline,
            AnimateDiffControlNetPipeline,
            AnimateDiffPAGPipeline,
            AnimateDiffPipeline,
            AnimateDiffSDXLPipeline,
            AnimateDiffSparseControlNetPipeline,
            AnimateDiffVideoToVideoControlNetPipeline,
            AnimateDiffVideoToVideoPipeline,
            AudioLDM2Pipeline,
            AudioLDM2ProjectionModel,
            AudioLDM2UNet2DConditionModel,
            AudioLDMPipeline,
            AuraFlowPipeline,
            BriaPipeline,
            ChromaImg2ImgPipeline,
            ChromaPipeline,
            CLIPImageProjection,
            CogVideoXFunControlPipeline,
            CogVideoXImageToVideoPipeline,
            CogVideoXPipeline,
            CogVideoXVideoToVideoPipeline,
            CogView3PlusPipeline,
            CogView4ControlPipeline,
            CogView4Pipeline,
            ConsisIDPipeline,
            Cosmos2TextToImagePipeline,
            Cosmos2VideoToWorldPipeline,
            CosmosTextToWorldPipeline,
            CosmosVideoToWorldPipeline,
            CycleDiffusionPipeline,
            EasyAnimateControlPipeline,
            EasyAnimateInpaintPipeline,
            EasyAnimatePipeline,
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
            HiDreamImagePipeline,
            HunyuanDiTControlNetPipeline,
            HunyuanDiTPAGPipeline,
            HunyuanDiTPipeline,
            HunyuanSkyreelsImageToVideoPipeline,
            HunyuanVideoFramepackPipeline,
            HunyuanVideoImageToVideoPipeline,
            HunyuanVideoPipeline,
            I2VGenXLPipeline,
            IFImg2ImgPipeline,
            IFImg2ImgSuperResolutionPipeline,
            IFInpaintingPipeline,
            IFInpaintingSuperResolutionPipeline,
            IFPipeline,
            IFSuperResolutionPipeline,
            ImageTextPipelineOutput,
            Kandinsky3Img2ImgPipeline,
            Kandinsky3Pipeline,
            KandinskyCombinedPipeline,
            KandinskyImg2ImgCombinedPipeline,
            KandinskyImg2ImgPipeline,
            KandinskyInpaintCombinedPipeline,
            KandinskyInpaintPipeline,
            KandinskyPipeline,
            KandinskyPriorPipeline,
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
            LatentConsistencyModelImg2ImgPipeline,
            LatentConsistencyModelPipeline,
            LattePipeline,
            LDMTextToImagePipeline,
            LEditsPPPipelineStableDiffusion,
            LEditsPPPipelineStableDiffusionXL,
            LTXConditionPipeline,
            LTXImageToVideoPipeline,
            LTXLatentUpsamplePipeline,
            LTXPipeline,
            LucyEditPipeline,
            Lumina2Pipeline,
            Lumina2Text2ImgPipeline,
            LuminaPipeline,
            LuminaText2ImgPipeline,
            MarigoldDepthPipeline,
            MarigoldIntrinsicsPipeline,
            MarigoldNormalsPipeline,
            MochiPipeline,
            MusicLDMPipeline,
            OmniGenPipeline,
            PaintByExamplePipeline,
            PIAPipeline,
            PixArtAlphaPipeline,
            PixArtSigmaPAGPipeline,
            PixArtSigmaPipeline,
            QwenImageControlNetInpaintPipeline,
            QwenImageControlNetPipeline,
            QwenImageEditInpaintPipeline,
            QwenImageEditPipeline,
            QwenImageEditPlusPipeline,
            QwenImageImg2ImgPipeline,
            QwenImageInpaintPipeline,
            QwenImagePipeline,
            ReduxImageEncoder,
            SanaControlNetPipeline,
            SanaPAGPipeline,
            SanaPipeline,
            SanaSprintImg2ImgPipeline,
            SanaSprintPipeline,
            SemanticStableDiffusionPipeline,
            ShapEImg2ImgPipeline,
            ShapEPipeline,
            SkyReelsV2DiffusionForcingImageToVideoPipeline,
            SkyReelsV2DiffusionForcingPipeline,
            SkyReelsV2DiffusionForcingVideoToVideoPipeline,
            SkyReelsV2ImageToVideoPipeline,
            SkyReelsV2Pipeline,
            StableAudioPipeline,
            StableAudioProjectionModel,
            StableCascadeCombinedPipeline,
            StableCascadeDecoderPipeline,
            StableCascadePriorPipeline,
            StableDiffusion3ControlNetInpaintingPipeline,
            StableDiffusion3ControlNetPipeline,
            StableDiffusion3Img2ImgPipeline,
            StableDiffusion3InpaintPipeline,
            StableDiffusion3PAGImg2ImgPipeline,
            StableDiffusion3PAGPipeline,
            StableDiffusion3Pipeline,
            StableDiffusionAdapterPipeline,
            StableDiffusionAttendAndExcitePipeline,
            StableDiffusionControlNetImg2ImgPipeline,
            StableDiffusionControlNetInpaintPipeline,
            StableDiffusionControlNetPAGInpaintPipeline,
            StableDiffusionControlNetPAGPipeline,
            StableDiffusionControlNetPipeline,
            StableDiffusionControlNetXSPipeline,
            StableDiffusionDepth2ImgPipeline,
            StableDiffusionDiffEditPipeline,
            StableDiffusionGLIGENPipeline,
            StableDiffusionGLIGENTextImagePipeline,
            StableDiffusionImageVariationPipeline,
            StableDiffusionImg2ImgPipeline,
            StableDiffusionInpaintPipeline,
            StableDiffusionInpaintPipelineLegacy,
            StableDiffusionInstructPix2PixPipeline,
            StableDiffusionLatentUpscalePipeline,
            StableDiffusionLDM3DPipeline,
            StableDiffusionModelEditingPipeline,
            StableDiffusionPAGImg2ImgPipeline,
            StableDiffusionPAGInpaintPipeline,
            StableDiffusionPAGPipeline,
            StableDiffusionPanoramaPipeline,
            StableDiffusionParadigmsPipeline,
            StableDiffusionPipeline,
            StableDiffusionPipelineSafe,
            StableDiffusionPix2PixZeroPipeline,
            StableDiffusionSAGPipeline,
            StableDiffusionUpscalePipeline,
            StableDiffusionXLAdapterPipeline,
            StableDiffusionXLControlNetImg2ImgPipeline,
            StableDiffusionXLControlNetInpaintPipeline,
            StableDiffusionXLControlNetPAGImg2ImgPipeline,
            StableDiffusionXLControlNetPAGPipeline,
            StableDiffusionXLControlNetPipeline,
            StableDiffusionXLControlNetUnionImg2ImgPipeline,
            StableDiffusionXLControlNetUnionInpaintPipeline,
            StableDiffusionXLControlNetUnionPipeline,
            StableDiffusionXLControlNetXSPipeline,
            StableDiffusionXLImg2ImgPipeline,
            StableDiffusionXLInpaintPipeline,
            StableDiffusionXLInstructPix2PixPipeline,
            StableDiffusionXLPAGImg2ImgPipeline,
            StableDiffusionXLPAGInpaintPipeline,
            StableDiffusionXLPAGPipeline,
            StableDiffusionXLPipeline,
            StableUnCLIPImg2ImgPipeline,
            StableUnCLIPPipeline,
            StableVideoDiffusionPipeline,
            TextToVideoSDPipeline,
            TextToVideoZeroPipeline,
            TextToVideoZeroSDXLPipeline,
            UnCLIPImageVariationPipeline,
            UnCLIPPipeline,
            UniDiffuserModel,
            UniDiffuserPipeline,
            UniDiffuserTextDecoder,
            VersatileDiffusionDualGuidedPipeline,
            VersatileDiffusionImageVariationPipeline,
            VersatileDiffusionPipeline,
            VersatileDiffusionTextToImagePipeline,
            VideoToVideoSDPipeline,
            VisualClozeGenerationPipeline,
            VisualClozePipeline,
            VQDiffusionPipeline,
            WanImageToVideoPipeline,
            WanPipeline,
            WanVACEPipeline,
            WanVideoToVideoPipeline,
            WuerstchenCombinedPipeline,
            WuerstchenDecoderPipeline,
            WuerstchenPriorPipeline,
        )

    try:
        if not (is_torch_available() and is_transformers_available() and is_k_diffusion_available()):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from .utils.dummy_torch_and_transformers_and_k_diffusion_objects import *  # noqa F403
    else:
        from .pipelines import StableDiffusionKDiffusionPipeline, StableDiffusionXLKDiffusionPipeline

    try:
        if not (is_torch_available() and is_transformers_available() and is_sentencepiece_available()):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from .utils.dummy_torch_and_transformers_and_sentencepiece_objects import *  # noqa F403
    else:
        from .pipelines import KolorsImg2ImgPipeline, KolorsPAGPipeline, KolorsPipeline

    try:
        if not (is_torch_available() and is_transformers_available() and is_opencv_available()):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from .utils.dummy_torch_and_transformers_and_opencv_objects import *  # noqa F403
    else:
        from .pipelines import ConsisIDPipeline

    try:
        if not (is_torch_available() and is_transformers_available() and is_onnx_available()):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from .utils.dummy_torch_and_transformers_and_onnx_objects import *  # noqa F403
    else:
        from .pipelines import (
            OnnxStableDiffusionImg2ImgPipeline,
            OnnxStableDiffusionInpaintPipeline,
            OnnxStableDiffusionInpaintPipelineLegacy,
            OnnxStableDiffusionPipeline,
            OnnxStableDiffusionUpscalePipeline,
            StableDiffusionOnnxPipeline,
        )

    try:
        if not (is_torch_available() and is_librosa_available()):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from .utils.dummy_torch_and_librosa_objects import *  # noqa F403
    else:
        from .pipelines import AudioDiffusionPipeline, Mel

    try:
        if not (is_transformers_available() and is_torch_available() and is_note_seq_available()):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from .utils.dummy_transformers_and_torch_and_note_seq_objects import *  # noqa F403
    else:
        from .pipelines import SpectrogramDiffusionPipeline

    try:
        if not is_flax_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from .utils.dummy_flax_objects import *  # noqa F403
    else:
        from .models.controlnets.controlnet_flax import FlaxControlNetModel
        from .models.modeling_flax_utils import FlaxModelMixin
        from .models.unets.unet_2d_condition_flax import FlaxUNet2DConditionModel
        from .models.vae_flax import FlaxAutoencoderKL
        from .pipelines import FlaxDiffusionPipeline
        from .schedulers import (
            FlaxDDIMScheduler,
            FlaxDDPMScheduler,
            FlaxDPMSolverMultistepScheduler,
            FlaxEulerDiscreteScheduler,
            FlaxKarrasVeScheduler,
            FlaxLMSDiscreteScheduler,
            FlaxPNDMScheduler,
            FlaxSchedulerMixin,
            FlaxScoreSdeVeScheduler,
        )

    try:
        if not (is_flax_available() and is_transformers_available()):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from .utils.dummy_flax_and_transformers_objects import *  # noqa F403
    else:
        from .pipelines import (
            FlaxStableDiffusionControlNetPipeline,
            FlaxStableDiffusionImg2ImgPipeline,
            FlaxStableDiffusionInpaintPipeline,
            FlaxStableDiffusionPipeline,
            FlaxStableDiffusionXLPipeline,
        )

    try:
        if not (is_note_seq_available()):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from .utils.dummy_note_seq_objects import *  # noqa F403
    else:
        from .pipelines import MidiProcessor

else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
        extra_objects={"__version__": __version__},
    )
