import dataclasses
from diffusers import AutoModel, LongCatImageTransformer2DModel

config_dict = AutoModel.load_config(
    "meituan-longcat/LongCat-Image",
    subfolder="transformer",
)
# import DiT based on _class_name
typed_config = LongCatImageTransformer2DModel._get_dataclass_from_config(config_dict)
for f in dataclasses.fields(typed_config):
    print(f"{f.name}: {f.type}")
