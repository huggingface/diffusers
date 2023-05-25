import torch

from diffusers import UnCLIPPipeline


pipe = UnCLIPPipeline.from_pretrained("kakaobrain/karlo-v1-alpha", torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt = "a high-resolution photograph of a big red frog on a green leaf."

image = pipe([prompt]).images[0]

image.save("./frog.png")
