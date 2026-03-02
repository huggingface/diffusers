import dataclasses
import torch
from diffusers import FluxTransformer2DModel
from diffusers.models import AutoModel

repo = "black-forest-labs/FLUX.2-dev"
subfolder = "transformer"

print("=== From load_config (no model instantiation) ===")
config_dict = FluxTransformer2DModel.load_config(repo, subfolder=subfolder)
tc = FluxTransformer2DModel._get_dataclass_from_config(config_dict)
print(f"Type: {type(tc).__name__}")
for k, v in dataclasses.asdict(tc).items():
    print(f"  {k}: {v}")

print()
print("=== From AutoModel.from_config on meta device ===")
with torch.device("meta"):
    model = AutoModel.from_config(repo, subfolder=subfolder)
print(f"model.config:")
for k, v in dict(model.config).items():
    if not k.startswith("_"):
        print(f"  {k}: {v}")

print()
print("=== Comparison ===")
dc_dict = dataclasses.asdict(tc)
config = {k: v for k, v in dict(model.config).items() if not k.startswith("_")}
print(f"Match: {dc_dict == config}")
