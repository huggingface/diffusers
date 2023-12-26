from diffusers.schedulers import DDIMScheduler

from pipeline_null_text_inversion import NullTextPipeline
import torch

invert_prompt = "A lying cat"
input_image = "siamese.jpg"
steps = 50


# if reconstruct 
prompt = "A lying cat"

# if edit
prompt = "A lying dog"


# must torch.float32
scheduler = DDIMScheduler(num_train_timesteps=1000, beta_start=0.00085, beta_end=0.0120, beta_schedule="scaled_linear")
pipeline = NullTextPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", scheduler = scheduler, torch_dtype=torch.float32).to("cuda")
inverted_latent, uncond = pipeline.invert(input_image, invert_prompt, num_inner_steps=5, early_stop_epsilon= 1e-5, num_inference_steps = steps)
pipeline(prompt, uncond, inverted_latent, guidance_scale=7.5, num_inference_steps=steps).images[0].save(input_image+".output.jpg")


