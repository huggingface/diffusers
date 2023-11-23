import torch
from PIL import Image
# from IPython.display import display
from diffusers import AutoPipelineForText2Image,AutoPipelineForImage2Image,StableDiffusionPipeline
from transformers import (
        CLIPImageProcessor,
        CLIPVisionModelWithProjection,
    )

from diffusers.utils import load_image
import debugpy

debugpy.listen(5678)
print("waiting")
debugpy.wait_for_client()
print("attach")

image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    "h94/IP-Adapter", 
    subfolder="models/image_encoder",
    torch_dtype=torch.float16,
).to("cuda")
# pipeline = AutoPipelineForText2Image.from_pretrained("runwayml/stable-diffusion-v1-5", image_encoder=image_encoder, torch_dtype=torch.float16).to("cuda")

# pipeline.set_ip_adapter_scale(0.6)
# image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/load_neg_embed.png")
# generator = torch.Generator(device="cpu").manual_seed(33)
# images = pipeline(
#     prompt='best quality, high quality, wearing sunglasses', 
#     ip_adapter_image=image,
#     negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality", 
#     num_inference_steps=50,
#     generator=generator,
# ).images
# new_img = images[0]
# new_img.save("Images/output.jpg")

# pipeline = AutoPipelineForImage2Image.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16).to("cuda")

# image = load_image("https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/vermeer.jpg")
# ip_image = load_image("https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/river.png")

# pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sdxl.bin")
# generator = torch.Generator(device="cpu").manual_seed(33)
# images = pipeline(
#     prompt="best quality, high quality", 
#     ip_adapter_image=image,
#     negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality", 
#     num_inference_steps=25,
#     generator=generator,
#     ).images
# new_img = images[0]
# new_img.save("Images/output.jpg")


base_model_path = "runwayml/stable-diffusion-v1-5"
pipeline = StableDiffusionPipeline.from_pretrained(
    base_model_path, image_encoder=image_encoder,torch_dtype=torch.float16)
pipeline.to("cuda")

image = load_image("https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/ai_face2.png")

pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin") #ip-adapter-full-face_sd15.bin
pipeline.set_ip_adapter_scale(0.6)
generator = torch.Generator(device="cpu").manual_seed(33)
images = pipeline(
    prompt="A photo of a girl wearing a black dress, holding red roses in hand, upper body, behind is the Eiffel Tower",
    ip_adapter_image=image,
    negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality", 
    num_inference_steps=50,
    generator=generator,
).images
new_img = images[0]
new_img.save("Images/output.jpg")