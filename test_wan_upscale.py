from PIL import Image
import requests
from io import BytesIO
from diffusers import WanLowNoiseUpscalePipeline

# Load a sample image from the web
url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"
response = requests.get(url)
image = Image.open(BytesIO(response.content))

# Initialize the pipeline
pipe = WanLowNoiseUpscalePipeline.from_pretrained()

# Test the upscale function
upscaled_image = pipe(
    image=image,
    prompt="a beautiful mountain landscape",
    scale=2.0,
    num_inference_steps=20,
    guidance_scale=1.2,
    strength=0.8
)

# Save the result
upscaled_image.save("upscaled_image.jpg")
print("Upscaled image saved as 'upscaled_image.jpg'")
