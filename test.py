import torch
from diffusers import Transformer3DModel

model = Transformer3DModel(
    in_channels=4,
    out_channels=8,
    cross_attention_dim=1408,
    num_embeds_ada_norm=1000,
    sample_size=(2, 4, 4),
)
channels, num_frames, height, width = 4, 2, 4, 4
x = torch.randn(1, channels, num_frames, height, width)
y = torch.randn(1, 77, 256)
t = torch.ones(1)

with torch.no_grad():
    out = model(x, y, t)
    print(out.sample.shape)  # torch.Size([1, 8, 2, 4, 4])