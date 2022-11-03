from .utils import (
    is_accelerate_available,
    is_flax_available,
    is_inflect_available,
    is_onnx_available,
    is_scipy_available,
    is_torch_available,
    is_transformers_available,
    is_unidecode_available,
)


__version__ = "0.7.0"

from .configuration_utils import ConfigMixin
from .onnx_utils import OnnxRuntimeModel
from .utils import logging


# This will create an extra dummy file "dummy_torch_and_accelerate_objects.py"
# TODO: (patil-suraj, anton-l) maybe import everything under is_torch_and_accelerate_available
if is_torch_available() and not is_accelerate_available():
    error_msg = "Please install the `accelerate` library to use Diffusers with PyTorch. You can do so by running `pip install diffusers[torch]`. Or if torch is already installed, you can run `pip install accelerate`."  # noqa: E501
    raise ImportError(error_msg)


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
        PNDMPipeline,
        RePaintPipeline,
        ScoreSdeVePipeline,
        VQDiffusionPipeline,
    )
    from .schedulers import (
        DDIMScheduler,
        DDPMScheduler,
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
        LDMTextToImagePipeline,
        StableDiffusionImg2ImgPipeline,
        StableDiffusionInpaintPipeline,
        StableDiffusionInpaintPipelineLegacy,
        StableDiffusionPipeline,
    )
else:
    from .utils.dummy_torch_and_transformers_objects import *  # noqa F403

if is_torch_available() and is_transformers_available() and is_onnx_available():
    from .pipelines import (
        OnnxStableDiffusionImg2ImgPipeline,
        OnnxStableDiffusionInpaintPipeline,
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
