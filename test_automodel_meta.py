import torch
from diffusers import AutoModel

repo = "meituan-longcat/LongCat-Image"
subfolder = "transformer"

config = AutoModel.load_config(repo, subfolder=subfolder)

with torch.device("meta"):
    model = AutoModel.from_config(config)
print(f"model.config:")
for k, v in dict(model.config).items():
    if not k.startswith("_"):
        print(f"  {k}: {v}")
