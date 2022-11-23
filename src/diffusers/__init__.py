from .utils import (
    is_flax_available,
    is_inflect_available,
    is_onnx_available,
    is_scipy_available,
    is_torch_available,
    is_transformers_available,
    is_unidecode_available,
)


__version__ = "0.8.0.dev0"

from .configuration_utils import ConfigMixin
from .onnx_utils import OnnxRuntimeModel
from .utils import logging


if is_torch_available():
    from .modeling_utils import ModelMixin
    from .models import AutoencoderKL, Transformer2DModel, UNet1DModel, UNet2DConditionModel, UNet2DModel, VQModel
    from .optimization import (
        get_constant_schedule,
        get_constant_schedule_with_warmup,
        get_cosine_schedule_with_warmup,
        get_cosine_with_hard_restarts_schedule_with_warmup,
        get_linear_schedule_with_warmup,
        get_polynomial_decay_schedule_with_warmup,
        get_scheduler,
    )
    from .pipeline_utils import DiffusionPipeline
    from .pipelines import (
        DanceDiffusionPipeline,
        DDIMPipeline,
        DDPMPipeline,
        KarrasVePipeline,
        LDMPipeline,
        LDMSuperResolutionPipeline,
        PNDMPipeline,
        RePaintPipeline,
        ScoreSdeVePipeline,
    )
    from .schedulers import (
        DDIMScheduler,
        DDPMScheduler,
        DPMSolverMultistepScheduler,
        EulerAncestralDiscreteScheduler,
        EulerDiscreteScheduler,
        IPNDMScheduler,
        KarrasVeScheduler,
        PNDMScheduler,
        RePaintScheduler,
        SchedulerMixin,
        ScoreSdeVeScheduler,
        VQDiffusionScheduler,
    )
    from .training_utils import EMAModel
else:
    from .utils.dummy_pt_objects import *  # noqa F403

if is_torch_available() and is_scipy_available():
    from .schedulers import LMSDiscreteScheduler
else:
    from .utils.dummy_torch_and_scipy_objects import *  # noqa F403

if is_torch_available() and is_transformers_available():
    from .pipelines import (
        AltDiffusionImg2ImgPipeline,
        AltDiffusionPipeline,
        CycleDiffusionPipeline,
        LDMTextToImagePipeline,
        StableDiffusionImg2ImgPipeline,
        StableDiffusionInpaintPipeline,
        StableDiffusionInpaintPipelineLegacy,
        StableDiffusionPipeline,
        StableDiffusionPipelineSafe,
        VQDiffusionPipeline,
    )
else:
    from .utils.dummy_torch_and_transformers_objects import *  # noqa F403

if is_torch_available() and is_transformers_available() and is_onnx_available():
    from .pipelines import (
        OnnxStableDiffusionImg2ImgPipeline,
        OnnxStableDiffusionInpaintPipeline,
        OnnxStableDiffusionInpaintPipelineLegacy,
        OnnxStableDiffusionPipeline,
        StableDiffusionOnnxPipeline,
    )
else:
    from .utils.dummy_torch_and_transformers_and_onnx_objects import *  # noqa F403

if is_flax_available():
    from .modeling_flax_utils import FlaxModelMixin
    from .models.unet_2d_condition_flax import FlaxUNet2DConditionModel
    from .models.vae_flax import FlaxAutoencoderKL
    from .pipeline_flax_utils import FlaxDiffusionPipeline
    from .schedulers import (
        FlaxDDIMScheduler,
        FlaxDDPMScheduler,
        FlaxDPMSolverMultistepScheduler,
        FlaxKarrasVeScheduler,
        FlaxLMSDiscreteScheduler,
        FlaxPNDMScheduler,
        FlaxSchedulerMixin,
        FlaxScoreSdeVeScheduler,
    )
else:
    from .utils.dummy_flax_objects import *  # noqa F403

if is_flax_available() and is_transformers_available():
    from .pipelines import FlaxStableDiffusionPipeline
else:
    from .utils.dummy_flax_and_transformers_objects import *  # noqa F403
