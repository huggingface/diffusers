import torch
from diffusers import AutoPipelineForText2Video
from diffusers.utils import export_to_video

wan_list = [
    "Wan-AI/Wan2.1-T2V-14B-Diffusers",
    "Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers",
    "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
    "Wan-AI/Wan2.1-VACE-1.3B-diffusers",
    "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
]

pipe = AutoPipelineForText2Video.from_pretrained(
    wan_list[3],
    #torch_dtype=torch.float16,
)

print(pipe.text_encoder.__class__.__name__)

# img = torch.randn(1, 3, 10, 512, 512)  # batch 1, 3 channels, 512x512
# latent = pipe.vae.encode(img).latent_dist.mode() # encoder output
# print("Latent shape:", latent.shape)  

# #Latent shape: torch.Size([1, 16, 3, 64, 64])

# recon =pipe.vae.decode(latent).sample
# print("Reconstructed image shape:", recon.shape)