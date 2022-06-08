import torch

from modeling_glide import GLIDE


generator = torch.Generator()
generator = generator.manual_seed(0)

# 1. Load models
pipeline = GLIDE.from_pretrained("fusing/glide-base")

img = pipeline("an oil painting of a corgi", generator)

print(img)
