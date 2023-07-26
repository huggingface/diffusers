from ...utils import OptionalDependencyNotAvailable, is_torch_available, is_transformers_available
from .pipeline_latent_diffusion_superresolution import LDMSuperResolutionPipeline


try:
    if not (is_transformers_available() and is_torch_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ...utils.dummy_torch_and_transformers_objects import ShapEPipeline
else:
    from .pipeline_latent_diffusion import LDMBertModel, LDMTextToImagePipeline
