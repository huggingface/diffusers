import torch

from modeling_glide import GLIDE
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['interactive'] = True


generator = torch.Generator()
generator = generator.manual_seed(0)

# 1. Load models
pipeline = GLIDE.from_pretrained("fusing/glide-base")

img = pipeline("a pencil sketch of a corgi", generator)
img = ((img + 1)*127.5).round().clamp(0, 255).to(torch.uint8).cpu().numpy()

plt.figure(figsize=(8, 8))
plt.imshow(img)
plt.show()
