import sys

sys.path.append("src/")
sys.path.append("src/diffusers/")

import torch
import numpy as np
import imageio
from diffusers import DiffusionPipeline


model_id = "CompVis/ldm-text2im-large-256"

seed = 0

# load model and scheduler
ldm = DiffusionPipeline.from_pretrained(model_id)



# run pipeline in inference (sample random noise and denoise)
prompt = "A squirrel eating a burger, trending on artstation"

embedding = ldm.embed_prompts([prompt])

image = ldm(text_embeddings=embedding, num_inference_steps=50, 
                eta=0.3, guidance_scale=6, seed=seed)["sample"][0]
image.save(f"squirrel.png")

squirrel_img = image
    
with torch.inference_mode():
    encoded_latents = ldm.encode_image(image)
    print("Encoded latents mean and std:", encoded_latents.mean(), encoded_latents.std())
    print("Encoded latents shape:", encoded_latents.shape)
    decoded_image = ldm.decode_image(encoded_latents)[0]
    decoded_image.save(f"decoded_squirrel.png")

    
# testing if we can continuously update an image
# run pipeline in inference (sample random noise and denoise)
prompt = "A cat eating a burger, trending on artstation"
embedding = ldm.embed_prompts([prompt])

import imageio

"""
for eta in np.linspace(0.1, 1, 10):
    eta_imgs = []
    print(eta)
    for steps in np.arange(1, 25, 1):
        steps = int(steps)
        image = ldm(text_embeddings=embedding, num_inference_steps=steps, 
                    eta=eta, guidance_scale=6, seed=seed, start_img=squirrel_img)["sample"][0]
        eta_imgs.append(image)
    imageio.mimsave(f"cat_eta{eta}.gif", eta_imgs)
"""

def minmax(a):
    return (a - a.min()) / (a.max() - a.min())

"""
# try adding noise of different intensities instead
for noise_strength in np.arange(0, 1.1, 0.1):
    noise_imgs = []
    print(noise_strength)
    for steps in np.arange(1, 25, 1):
        steps = int(steps)
        image = ldm(text_embeddings=embedding, num_inference_steps=steps, noise_strength=noise_strength,
                 guidance_scale=6, seed=seed, start_img=squirrel_img)["sample"][0]
        noise_imgs.append(image)
    name = f"cat_noise_strength{noise_strength:.2f}.gif"
    imageio.mimsave(name, noise_imgs)
    print("Saved at ", name)
"""
    
"""
# try adding noise of different time_steps instead
for noise_step in np.arange(1, 1000, 100):
    noise_imgs = []
    print(noise_step)
    for steps in np.arange(1, 25, 1):
        steps = int(steps)
        image = ldm(text_embeddings=embedding, num_inference_steps=steps, noise_step=noise_step,
                 guidance_scale=6, seed=seed, start_img=squirrel_img)["sample"][0]
        noise_imgs.append(image)
    name = f"cat_noise_step{noise_step}.gif"
    imageio.mimsave(name, noise_imgs)
    print("Saved at ", name)
"""

# with chosen noise strength, test different etas at differnet steps
chosen_noise_strength = 0.8
for eta in np.linspace(0.0, 1, 10):
    eta_imgs = []
    print(eta)
    for steps in np.arange(1, 25, 1):
        steps = int(steps)
        image = ldm(text_embeddings=embedding, num_inference_steps=steps, 
                    eta=eta, guidance_scale=6, seed=seed, start_img=squirrel_img, 
                    noise_strength=chosen_noise_strength)["sample"][0]
        eta_imgs.append(image)
    imageio.mimsave(f"cat_strength{chosen_noise_strength:.1f}_eta{eta:.1f}.gif", eta_imgs)

    
# with chosen noise step, test different etas at differnet steps
chosen_noise_step = 600
for eta in np.linspace(0.0, 1, 10):
    eta_imgs = []
    print(eta)
    for steps in np.arange(1, 25, 1):
        steps = int(steps)
        image = ldm(text_embeddings=embedding, num_inference_steps=steps, 
                    eta=eta, guidance_scale=6, seed=seed, start_img=squirrel_img, 
                    noise_step=chosen_noise_step)["sample"][0]
        eta_imgs.append(image)
    imageio.mimsave(f"cat_step{chosen_noise_step}_eta{eta:.1f}.gif", eta_imgs)
    
# testing if we can morph between two images in image latent space (yes we can!)
prompt_1 = "Cyberpunk, dystopian future, digital artwork"
prompt_2 = "Blissful, happy future, digital artwork"
embed_1 = ldm.embed_prompts([prompt_1])
embed_2 = ldm.embed_prompts([prompt_2])
latent_1 = ldm(text_embeddings=embed_1, num_inference_steps=50, eta=0.3, guidance_scale=6, seed=seed,
              output_type="latent")["sample"]
latent_2 = ldm(text_embeddings=embed_2, num_inference_steps=50, eta=0.3, guidance_scale=6, seed=seed, output_type="latent")["sample"]
# do interpolation
interpols = []
with torch.inference_mode():
    for interpol_weight in np.linspace(0, 1, 60):
        latent_mixed = latent_1 * (1 - interpol_weight) + latent_2 * interpol_weight
        image = ldm.decode_image(latent_mixed, output_type="pil")[0]
        interpols.append(image)
imageio.mimsave(f"interpol_digital.gif", interpols)
    

# testing interpolation in prompt latent spec
prompt = "A serene landscape with a river"
neg_prompts = "A serene landscape with a river. It feels fast"
embedding = ldm.embed_prompts([prompt])
neg_embedding = ldm.embed_prompts([neg_prompts])

print("Embedding shape: ", embedding.shape)
all_imgs = []
for neg_weights in np.linspace(0, 1, 24):
    merged_embedding = embedding * (1 - neg_weights) + neg_embedding * neg_weights
    images = ldm(text_embeddings=merged_embedding, num_inference_steps=50,
                eta=0.3, guidance_scale=6, seed=seed)["sample"]

    # save images
    images[0].save(f"squirrel-{0}-{neg_weights}.png")        
    all_imgs.append(images[0])
    
        
import imageio
imageio.mimsave("river_scary_simple.gif", all_imgs)
