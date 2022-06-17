# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.
from .utils import is_transformers_available

__version__ = "0.0.4"

from .modeling_utils import ModelMixin
from .models.unet import UNetModel
from .models.unet_grad_tts import UNetGradTTSModel
from .models.unet_ldm import UNetLDMModel
from .pipeline_utils import DiffusionPipeline
from .pipelines import BDDM, DDIM, DDPM, GLIDE, PNDM, GradTTS, LatentDiffusion
from .schedulers import DDIMScheduler, DDPMScheduler, GradTTSScheduler, PNDMScheduler, SchedulerMixin
from .schedulers.classifier_free_guidance import ClassifierFreeGuidanceScheduler

if is_transformers_available():
    from .models.unet_glide import GLIDESuperResUNetModel, GLIDETextToImageUNetModel, GLIDEUNetModel
else:
    from .utils.dummy_transformers_objects import *
