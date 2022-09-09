# --------------------------------------------------------------------------------
# diffusers/src

from diffusers import UNet2DModel

repo_id = "google/ddpm-church-256"
model = UNet2DModel.from_pretrained(repo_id)
print(type(model.config))

config = dict(model.config)
config["block_out_channels"] = [
    256,
    256,
    512,
    512,
    1024,
    1024,
  ]
config["resnet_time_scale_shift"] = "scale_shift"
model = UNet2DModel(**config)

print(model)
model.save_pretrained("ddpm-church-256-custom")

config = dict(model.config)
print(config)
print(config["resnet_time_scale_shift"])

# --------------------------------------------------------------------------------
# diffusers/src

import torch

ckpt = "ddpm-church-256-custom/diffusion_pytorch_model.bin"
model = torch.load(ckpt)

# print(model)
# print(model.keys())

# for k in model.keys():
#     if k.startswith("time_"):
#         print(k)

info = {k: str(list(v.shape)) for k, v in model.items()}
import json
with open("target.json", "w") as fp:
    json.dump(info, fp, indent=4)

# --------------------------------------------------------------------------------
# diffusers/src

import torch

ckpt = "checkpoints/256x256_diffusion_uncond.pt"
model = torch.load(ckpt)
# print(model)
# print(model.keys())

# for k in model.keys():
#    if k.startswith("time_embed"):
#        print(k)

info = {k: str(list(v.shape)) for k, v in model.items()}
import json
with open("guided-diffusion.json", "w") as fp:
    json.dump(info, fp, indent=4)
