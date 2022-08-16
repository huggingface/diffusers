# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.
from .utils import is_inflect_available, is_scipy_available, is_transformers_available, is_unidecode_available


__version__ = "0.1.3"

from .modeling_utils import ModelMixin
from .models import AutoencoderKL, UNet2DConditionModel, UNet2DModel, VQModel
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
from .pipelines import DDIMPipeline, DDPMPipeline, KarrasVePipeline, LDMPipeline, PNDMPipeline, ScoreSdeVePipeline
from .schedulers import (
    DDIMScheduler,
    DDPMScheduler,
    KarrasVeScheduler,
    PNDMScheduler,
    SchedulerMixin,
    ScoreSdeVeScheduler,
)


if is_scipy_available():
    from .schedulers import LMSDiscreteScheduler

from .training_utils import EMAModel


if is_transformers_available():
    from .pipelines import LDMTextToImagePipeline, StableDiffusionPipeline


else:
    from .utils.dummy_transformers_objects import *
