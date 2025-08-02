from diffusers import AutoencoderKLMMQuant
import torch

model = AutoencoderKLMMQuant.from_pretrained("onkarsus13/MMVQVae", subfolder="vae", torch_dtype=torch.float32)

