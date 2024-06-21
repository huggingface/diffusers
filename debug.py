from diffusers import LuminaText2ImgPipeline
import torch 

pipeline = LuminaText2ImgPipeline.from_pretrained(
	"/mnt/hdd1/xiejunlin/checkpoints/Lumina-Next-SFT-diffusers", torch_dtype=torch.bfloat16
).to("cuda")

image = pipeline(prompt="Upper body of a young woman in a Victorian-era outfit with brass goggles and leather straps. Background shows an industrial revolution cityscape with smoky skies and tall, metal structures").images[0]