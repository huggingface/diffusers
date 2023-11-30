from ..pipeline_utils import DiffusionPipeline
from .pipeline_output import LEditsPPDiffusionPipelineOutput, LEditsPPInversionPipelineOutput
from .ledits_utils import *

class LEditsPPPipelineStableDiffusion(DiffusionPipeline):
    """
    Pipeline for textual image editing using LEDits++ with Stable Diffusion.

    This model inherits from [`DiffusionPipeline`] and builds on the [`StableDiffusionPipeline`]. Check the superclass
    documentation for the generic methods implemented for all pipelines (downloading, saving, running on a particular
    device, etc.).
    """
    pass