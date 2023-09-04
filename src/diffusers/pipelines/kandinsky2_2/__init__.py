from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    get_objects_from_module,
    is_torch_available,
    is_transformers_available,
)


_import_structure = {}
_dummy_objects = {}


try:
    if not (is_transformers_available() and is_torch_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ...utils import dummy_torch_and_transformers_objects  # noqa F403

    _dummy_objects.update(get_objects_from_module(dummy_torch_and_transformers_objects))

else:
    _import_structure["pipeline_kandinsky2_2"] = ["KandinskyV22Pipeline"]
    _import_structure["pipeline_kandinsky2_2_combined"] = [
        "KandinskyV22CombinedPipeline",
        "KandinskyV22Img2ImgCombinedPipeline",
        "KandinskyV22InpaintCombinedPipeline",
    ]
    _import_structure["pipeline_kandinsky2_2_controlnet"] = ["KandinskyV22ControlnetPipeline"]
    _import_structure["pipeline_kandinsky2_2_controlnet_img2img"] = ["KandinskyV22ControlnetImg2ImgPipeline"]
    _import_structure["pipeline_kandinsky2_2_img2img"] = ["KandinskyV22Img2ImgPipeline"]
    _import_structure["pipeline_kandinsky2_2_inpainting"] = ["KandinskyV22InpaintPipeline"]
    _import_structure["pipeline_kandinsky2_2_prior"] = ["KandinskyV22PriorPipeline"]
    _import_structure["pipeline_kandinsky2_2_prior_emb2emb"] = ["KandinskyV22PriorEmb2EmbPipeline"]


import sys


sys.modules[__name__] = _LazyModule(
    __name__,
    globals()["__file__"],
    _import_structure,
    module_spec=__spec__,
)

for name, value in _dummy_objects.items():
    setattr(sys.modules[__name__], name, value)
