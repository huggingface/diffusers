from ...utils import (
    OptionalDependencyNotAvailable,
    is_torch_available,
    is_transformers_available,
    is_transformers_version,
)


try:
    if not (is_transformers_available() and is_torch_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ...utils.dummy_torch_and_transformers_objects import ShapEPipeline
else:
    from .pipeline_shap_e import ShapEPipeline
    from .params_proj import ShapEParamsProjModel
    from .renderer import MLPNeRSTFModel, MLPNeRFModelOutput, VoidNeRFModel, BoundingBoxVolume, StratifiedRaySampler, ImportanceRaySampler
    from .camera import create_pan_cameras
