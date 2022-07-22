# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.
from .utils import is_inflect_available, is_transformers_available, is_unidecode_available


__version__ = "0.1.2"

from .modeling_utils import ModelMixin
from .models import AutoencoderKL, UNet2DConditionModel, UNet2DModel, VQModel
from .optimization import (
    get_constant_schedule,
    get_constant_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
    get_scheduler,
)
from .pipeline_utils import DiffusionPipeline
from .pipelines import DDIMPipeline, DDPMPipeline, LDMPipeline, PNDMPipeline, ScoreSdeVePipeline
from .schedulers import DDIMScheduler, DDPMScheduler, PNDMScheduler, SchedulerMixin, ScoreSdeVeScheduler
from .training_utils import EMAModel


if is_transformers_available():
    from .pipelines import LDMTextToImagePipeline
else:
    from .utils.dummy_transformers_objects import *
