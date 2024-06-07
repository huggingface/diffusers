from diffusers import PixArtTransformer2DModel

ckpt_id = "https://huggingface.co/PixArt-alpha/PixArt-Sigma/blob/main/PixArt-Sigma-XL-2-1024-MS.pth"
model = PixArtTransformer2DModel.from_single_file(ckpt_id, original_config=True)