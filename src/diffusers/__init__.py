__version__ = "0.37.0.dev0"

from typing import TYPE_CHECKING

from .utils import (
    DIFFUSERS_SLOW_IMPORT,
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_flax_available,
    is_scipy_available,
    is_torch_available,
    is_torchsde_available,
    is_transformers_available,
    is_transformers_version,
)


# Lazy Import based on
# https://github.com/huggingface/transformers/blob/main/src/transformers/__init__.py

# When adding a new object to this init, please add it to `_import_structure`. The `_import_structure` is a dictionary submodule to list of object names,
# and is used to defer the actual importing for when the objects are requested.
# This way `import diffusers` provides the names in the namespace without actually importing anything (and especially none of the backends).

_import_structure = {
    "configuration_utils": ["ConfigMixin"],
    "models": [],
    "pipelines": [],
    "schedulers": [],
    "utils": [
        "OptionalDependencyNotAvailable",
        "is_flax_available",
        "is_scipy_available",
        "is_torch_available",
        "is_torchsde_available",
        "is_transformers_available",
        "is_transformers_version",
        "logging",
    ],
}

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils import dummy_pt_objects  # noqa F403

    _import_structure["utils.dummy_pt_objects"] = [name for name in dir(dummy_pt_objects) if not name.startswith("_")]

else:
    _import_structure["models"].extend(
        [
            "AutoencoderKL",
            "ModelMixin",
            "Transformer2DModel",
            "UNet2DConditionModel",
            "UNet2DModel",
            "VQModel",
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
            "DDPMPipeline",
            "DiffusionPipeline",
            "ImagePipelineOutput",
        ]
    )
    _import_structure["schedulers"].extend(
        [
            "DDIMScheduler",
            "DDPMScheduler",
            "EulerDiscreteScheduler",
            "PNDMScheduler",
            "SchedulerMixin",
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
    _import_structure["schedulers"].extend(["DPMSolverSDEScheduler"])

try:
    if not (is_torch_available() and is_transformers_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils import dummy_torch_and_transformers_objects  # noqa F403

    _import_structure["utils.dummy_torch_and_transformers_objects"] = [
        name for name in dir(dummy_torch_and_transformers_objects) if not name.startswith("_")
    ]

else:
    _import_structure["pipelines"].extend(
        [
            "LDMPipeline",
            "LDMSuperResolutionPipeline",
            "LDMTextToImagePipeline",
            "StableDiffusionControlNetImg2ImgPipeline",
            "StableDiffusionControlNetInpaintPipeline",
            "StableDiffusionControlNetPipeline",
            "StableDiffusionDepth2ImgPipeline",
            "StableDiffusionImageVariationPipeline",
            "StableDiffusionImg2ImgPipeline",
            "StableDiffusionInpaintPipeline",
            "StableDiffusionInpaintPipelineLegacy",
            "StableDiffusionInstructPix2PixPipeline",
            "StableDiffusionLatentUpscalePipeline",
            "StableDiffusionPipeline",
            "StableDiffusionUpscalePipeline",
        ]
    )


try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils import dummy_flax_objects  # noqa F403

    _import_structure["utils.dummy_flax_objects"] = [
        name for name in dir(dummy_flax_objects) if not name.startswith("_")
    ]


else:
    _import_structure["models.modeling_flax_utils"] = ["FlaxModelMixin"]
    _import_structure["models.unets.unet_2d_condition_flax"] = ["FlaxUNet2DConditionModel"]
    _import_structure["models.vae_flax"] = ["FlaxAutoencoderKL"]
    _import_structure["pipelines"].extend(["FlaxDiffusionPipeline"])
    _import_structure["schedulers"].extend(
        [
            "FlaxDDIMScheduler",
            "FlaxDDPMScheduler",
            "FlaxEulerDiscreteScheduler",
            "FlaxPNDMScheduler",
            "FlaxSchedulerMixin",
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
        ]
    )

if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    from .configuration_utils import ConfigMixin

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from .utils.dummy_pt_objects import *  # noqa F403
    else:
        from .models import (
            AutoencoderKL,
            ModelMixin,
            Transformer2DModel,
            UNet2DConditionModel,
            UNet2DModel,
            VQModel,
        )
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
            DDPMPipeline,
            DiffusionPipeline,
            ImagePipelineOutput,
        )
        from .schedulers import (
            DDIMScheduler,
            DDPMScheduler,
            EulerDiscreteScheduler,
            PNDMScheduler,
            SchedulerMixin,
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
        from .schedulers import DPMSolverSDEScheduler

    try:
        if not (is_torch_available() and is_transformers_available()):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from .utils.dummy_torch_and_transformers_objects import *  # noqa F403
    else:
        from .pipelines import (
            LDMPipeline,
            LDMSuperResolutionPipeline,
            LDMTextToImagePipeline,
            StableDiffusionControlNetImg2ImgPipeline,
            StableDiffusionControlNetInpaintPipeline,
            StableDiffusionControlNetPipeline,
            StableDiffusionDepth2ImgPipeline,
            StableDiffusionImageVariationPipeline,
            StableDiffusionImg2ImgPipeline,
            StableDiffusionInpaintPipeline,
            StableDiffusionInpaintPipelineLegacy,
            StableDiffusionInstructPix2PixPipeline,
            StableDiffusionLatentUpscalePipeline,
            StableDiffusionPipeline,
            StableDiffusionUpscalePipeline,
        )

    try:
        if not is_flax_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from .utils.dummy_flax_objects import *  # noqa F403
    else:
        from .models.modeling_flax_utils import FlaxModelMixin
        from .models.unets.unet_2d_condition_flax import FlaxUNet2DConditionModel
        from .models.vae_flax import FlaxAutoencoderKL
        from .pipelines import FlaxDiffusionPipeline
        from .schedulers import (
            FlaxDDIMScheduler,
            FlaxDDPMScheduler,
            FlaxEulerDiscreteScheduler,
            FlaxPNDMScheduler,
            FlaxSchedulerMixin,
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
        )

else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
        extra_objects={"__version__": __version__},
    )
