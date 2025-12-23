import json
import os

import torch
from diffusers import BriaFiboPipeline, BriaFiboEditPipeline
from diffusers.modular_pipelines import ModularPipeline

# -------------------------------
# Load the VLM pipeline
# -------------------------------
torch.set_grad_enabled(False)
# Using Gemini API, requires GOOGLE_API_KEY environment variable
assert os.getenv("GOOGLE_API_KEY") is not None, "GOOGLE_API_KEY environment variable is not set"
vlm_pipe = ModularPipeline.from_pretrained("briaai/FIBO-gemini-prompt-to-JSON", trust_remote_code=True)


pipe = BriaFiboPipeline.from_pretrained(
    "briaai/FIBO",
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")

output = vlm_pipe(
    prompt="A hyper-detailed, ultra-fluffy owl sitting in the trees at night, looking directly at the camera with wide, adorable, expressive eyes. Its feathers are soft and voluminous, catching the cool moonlight with subtle silver highlights. The owl's gaze is curious and full of charm, giving it a whimsical, storybook-like personality."
)
json_prompt_generate = output.values["json_prompt"]

results_generate = pipe(
    prompt=json_prompt_generate, num_inference_steps=50, guidance_scale=5
)
results_generate.images[0].save("image_generate.png")