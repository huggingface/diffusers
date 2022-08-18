# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.
from .utils import is_inflect_available, is_scipy_available, is_transformers_available, is_unidecode_available


__version__ = "0.2.2"

from .modeling_utils import ModelMixin

from .models.unet_2d import UNet2DModel
from .models.unet_2d_condition import UNet2DConditionModel
from .models.vae import AutoencoderKL, VQModel

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
from .pipelines.ddim import DDIMPipeline
from .pipelines.ddpm import DDPMPipeline
from .pipelines.latent_diffusion_uncond import LDMPipeline
from .pipelines.pndm import PNDMPipeline
from .pipelines.score_sde_ve import ScoreSdeVePipeline

from .pipelines.stochatic_karras_ve import KarrasVePipeline

if is_transformers_available():
    from .pipelines.latent_diffusion import LDMTextToImagePipeline
    from .pipelines.stable_diffusion import StableDiffusionPipeline
else:
    from .utils.dummy_transformers_objects import *

from .schedulers.ddim import DDIMScheduler
from .schedulers.ddpm import DDPMScheduler
from .schedulers.karras_ve import KarrasVeScheduler
from .schedulers.pndm import PNDMScheduler
from .schedulers.sde_ve import ScoreSdeVeScheduler
from .schedulers.sde_vp import ScoreSdeVpScheduler

from .schedulers.scheduling_utils import SchedulerMixin

if is_scipy_available():
    from .schedulers.lms_discrete import LMSDiscreteScheduler
else:
    from .utils.dummy_scipy_objects import *

from .training_utils import EMAModel


if is_transformers_available():
    from .pipelines import LDMTextToImagePipeline, StableDiffusionPipeline
else:
    from .utils.dummy_transformers_objects import *
