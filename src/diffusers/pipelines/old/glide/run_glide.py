import torch

import PIL.Image
from diffusers import DiffusionPipeline


generator = torch.Generator()
generator = generator.manual_seed(0)

model_id = "fusing/glide-base"

# load model and scheduler
pipeline = DiffusionPipeline.from_pretrained(model_id)

# run inference (text-conditioned denoising + upscaling)
img = pipeline("a crayon drawing of a corgi", generator)

# process image to PIL
img = img.squeeze(0)
img = ((img + 1) * 127.5).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
image_pil = PIL.Image.fromarray(img)

# save image
image_pil.save("test.png")
