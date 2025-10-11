from typing import TYPE_CHECKING

from .pipeline_output import PhotonPipelineOutput
from .pipeline_photon import PhotonPipeline


__all__ = ["PhotonPipeline", "PhotonPipelineOutput"]

# Make T5GemmaEncoder importable from this module for pipeline loading
if TYPE_CHECKING:
    from transformers.models.t5gemma.modeling_t5gemma import T5GemmaEncoder
else:
    try:
        from transformers.models.t5gemma.modeling_t5gemma import T5GemmaEncoder
    except ImportError:
        pass
