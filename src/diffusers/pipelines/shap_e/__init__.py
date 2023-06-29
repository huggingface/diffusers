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
    from .camera import create_pan_cameras
    from .pipeline_shap_e import ShapEPipeline
    from .pipeline_shap_e_img2img import ShapEImg2ImgPipeline
    from .renderer import (
        BoundingBoxVolume,
        ImportanceRaySampler,
        MLPNeRFModelOutput,
        MLPNeRSTFModel,
        ShapEParamsProjModel,
        ShapERenderer,
        StratifiedRaySampler,
        VoidNeRFModel,
    )
