import sys
import os
sys.path.append("src/")
sys.path.append("src/diffusers/")

# make sure you're logged in with `huggingface-cli login`
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler


os.makedirs("stable_outs", exist_ok=True)

lms = LMSDiscreteScheduler(
    beta_start=0.00085, 
    beta_end=0.012, 
    beta_schedule="scaled_linear"
)

pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4", 
    scheduler=lms,
    use_auth_token=True,
    revision="fp16",
    torch_dtype=torch.float16,
).to("cuda")

def pathify_prompt(prompt):
    return prompt.replace(" ", "_").replace(",", "")

def save_image(image, prompt):
    path = f"stable_outs/{pathify_prompt(prompt)}.jpg"
    image.save(path, quality=95, subsample=False)
    

seed = 0
prompt_list = ["Neural dust penetrates the mind and soul", 
               "Neural dust penetrates the mind and soul. A masterpiece", 
               "Neural dust penetrates the mind and soul. Masterpiece by Anton Wiehe", 
               "My soul transcends the ashes of my body to merge with the blissful, dimensionless void",
               "My soul transcends the ashes of my body to merge with the blissful, dimensionless void. A masterpiece",
               "My soul transcends the ashes of my body to merge with the blissful, dimensionless void. Masterpiece by Anton Wiehe",
               ]
prompt_list = []
with autocast("cuda"):
    for prompt in prompt_list:
        print(prompt)
        image = pipe(prompt, output_type="pil", use_safety=False,
                     num_inference_steps=100, eta=0.2, guidance_scale=7.5, seed=seed,
                     width=512, height=512, start_img=None,
                     )["sample"][0]  
        save_image(image, prompt)
        
        
import numpy as np
import imageio

prompt = "My soul transcends the ashes of my body to merge with the blissful, dimensionless void. Masterpiece by Anton Wiehe"
with autocast("cuda"):
    masterpiece_image = pipe(prompt, output_type="pil", use_safety=False,
                        num_inference_steps=50, eta=0.2, guidance_scale=7.5, seed=0,
                        width=512, height=512, start_img=None,
                        )["sample"][0]  
masterpiece_image.save("stable_outs/masterpiece.jpg", quality=95, subsample=False)
        
"""
# try adding noise of different time_steps
for noise_step in np.arange(400, 1000, 100):
    noise_imgs = []
    print(noise_step)
    for steps in np.arange(1, 25, 1):
        steps = int(steps)
        with autocast("cuda"):
            image = pipe(prompt="A cat eating a burger", num_inference_steps=steps, 
                         noise_step=noise_step, use_safety=False,
                    guidance_scale=6, seed=seed, start_img=masterpiece_image)["sample"][0]
        noise_imgs.append(image)
    name = f"masterpiece_noise_step{noise_step}.gif"
    path = os.path.join("stable_gifs", name)
    os.makedirs(path, exist_ok=True)
    imageio.mimsave(path, noise_imgs)
    print("Saved at ", path)
"""


"""
# try approach of diffusers img2img - does not work yet
for strength in np.arange(0.1, 1.0, 0.1):
    noise_imgs = []
    print(strength)
    for steps in np.arange(4, 15, 1):
        steps = int(steps)
        with autocast("cuda"):
            image = pipe(prompt="A cat eating a burger", num_inference_steps=steps, 
                         img2img_strength=strength, use_safety=False,
                    guidance_scale=6, seed=seed, start_img=masterpiece_image)["sample"][0]
        noise_imgs.append(image)
    name = f"masterpiece_img2img_strength{strength}.gif"
    path = os.path.join("stable_gifs", name)
    os.makedirs("stable_gifs", exist_ok=True)
    imageio.mimsave(path, noise_imgs)
    print("Saved at ", path)
"""
    
    
# try dynamic thresholding 
imgs = []
for threshold in [0.0, 0.5, 0.9, 0.95, 0.99, 0.995, 0.999]:
    with autocast("cuda"):
        image = pipe(prompt="A cat eating a burger", num_inference_steps=50, 
                        use_safety=False, dynamic_thresholding_quant=threshold,
                guidance_scale=6, seed=seed)["sample"][0]
    imgs.append(image)

imageio.mimsave("stable_gifs/cat_dynamic_thresholding.gif", imgs)