from diffusers import FabricPipeline
import torch

model_id = "dreamlike-art/dreamlike-photoreal-2.0"
pipe = FabricPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt = "photo, a church in the middle of a field of crops, bright cinematic lighting, gopro, fisheye lens"
image = pipe(prompt, n_images=1).images[0]

image.save("result.jpg")


