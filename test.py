from diffusers import AutoencoderKLMMQuant
import torch

model = AutoencoderKLMMQuant.from_pretrained("Wan-AI/Wan2.2-T2V-A14B-Diffusers", subfolder="vae", torch_dtype=torch.float32, cache_dir="/data2/onkar/video_diffusion_weights/VAE")

print(model)
