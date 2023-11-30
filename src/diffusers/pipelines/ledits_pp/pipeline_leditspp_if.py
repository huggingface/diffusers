from ..pipeline_utils import DiffusionPipeline
from .pipeline_output import LEditsPPDiffusionPipelineOutput, LEditsPPInversionPipelineOutput
from .ledits_utils import *

class LEditsPPPipelineIF(DiffusionPipeline):
    """
    Pipeline for textual image editing using LEDits++ with DeepfloydIF.

    This model inherits from [`DiffusionPipeline`] and builds on the [`IFPipeline`]. Check the superclass
    documentation for the generic methods implemented for all pipelines (downloading, saving, running on a particular
    device, etc.).
    """
    pass