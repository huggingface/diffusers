from diffusers import StableDiffusionPipeline
import torch
import os

model_id = "/scratch/mp5847/diffusers_ckpt/textual_inversion/van_gogh_50_attention_lr=5.0e-04_sd_v1.4_esd"
pipe = StableDiffusionPipeline.from_pretrained(model_id,torch_dtype=torch.float16).to("cuda")

prompt = "New York City in the style of <art-style>"
# prompt = "The Starry Night by <art-style>"

for i in range(10):
    image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]

    output_dir = "/scratch/mp5847/diffusers_test/textual_inversion/van_gogh_50_attention_lr=5.0e-04_sd_v1.4_esd"
    os.makedirs(output_dir, exist_ok=True)
    image.save(os.path.join(output_dir, f"test_{i}.png"))