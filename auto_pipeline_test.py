import torch
from diffusers import AutoPipelineForText2Video
from diffusers.utils import export_to_video

pipe = AutoPipelineForText2Video.from_pretrained(
    "THUDM/CogVideoX-5b",
    torch_dtype=torch.bfloat16,
)
