# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.
from .utils import is_inflect_available, is_transformers_available, is_unidecode_available


__version__ = "0.0.4"

from .modeling_utils import ModelMixin
from .models import AutoencoderKL, NCSNpp, UNetLDMModel, UNetModel, UNetUnconditionalModel, VQModel
from .pipeline_utils import DiffusionPipeline
from .pipelines import (
    DDIMPipeline,
    DDPMPipeline,
    LatentDiffusionUncondPipeline,
    PNDMPipeline,
    ScoreSdeVePipeline,
    ScoreSdeVpPipeline,
)
from .schedulers import (
    DDIMScheduler,
    DDPMScheduler,
    PNDMScheduler,
    SchedulerMixin,
    ScoreSdeVeScheduler,
    ScoreSdeVpScheduler,
)


if is_transformers_available():
    from .models.unet_glide import GlideSuperResUNetModel, GlideTextToImageUNetModel, GlideUNetModel
    from .pipelines import GlidePipeline, LatentDiffusionPipeline
else:
    from .utils.dummy_transformers_objects import *
