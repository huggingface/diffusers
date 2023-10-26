# from diffusers import StableDiffusionLDM3DPipeline
# import torch

# # pipe = StableDiffusionLDM3DPipeline.from_pretrained("Intel/ldm3d-pano") #, torch_dtype= torch.float16)
# pipe = StableDiffusionLDM3DPipeline.from_pretrained("Intel/ldm3d-pano") #, torch_dtype= torch.float16)

# pipe.to("cuda")
# prompt =  "360 view of a large bedroom"
# output = pipe(
#         prompt,
#         width=1024,
#         height=512,
#         guidance_scale=7.0,
#         num_inference_steps=50,
#     ) 

# rgb_image, depth_image = output.rgb, output.depth
# rgb_image[0].save("360_ldm3d_rgb2.jpg")
# depth_image[0].save("360_ldm3d_depth2.png")


from PIL import Image
from diffusers import StableDiffusionUpscalePipeline
import torch
import cv2
# # load model and scheduler
# model_id = "stabilityai/stable-diffusion-x4-upscaler"
# pipeline = StableDiffusionUpscalePipeline.from_pretrained(model_id, torch_dtype=torch.float16)
# pipeline = pipeline.to("cuda")

# low_res_img = Image.open("360_ldm3d_rgb2.jpg").convert("RGB")
# # low_res_img = low_res_img.resize((128, 128))


# upscaled_image = pipeline(prompt=prompt, image=low_res_img).images[0]
# upscaled_image.save("upsampled_360_ldm3d_rgb2.png")


# image = cv2.imread("360_ldm3d_rgb2.jpg")
# print("Size of image before pyrUp: ", image.shape)

# image = cv2.pyrUp(image)
# print("Size of image after pyrUp: ", image.shape)

# cv2.imwrite("upsampled_360_ldm3d_rgb2.png", image)


import gradio as gr
import requests
from PIL import Image
import os
import torch
import numpy as np
from transformers import AutoImageProcessor, Swin2SRForImageSuperResolution


processor = AutoImageProcessor.from_pretrained("caidas/swin2SR-classical-sr-x2-64")
model = Swin2SRForImageSuperResolution.from_pretrained("caidas/swin2SR-classical-sr-x2-64")

def enhance(image_path):
    image = Image.open(image_path)
    # prepare image for the model
    inputs = processor(image, return_tensors="pt")
    # forward pass
    print("Forward pass")
    with torch.no_grad():
        outputs = model(**inputs)
    print("Done")
    # postprocess
    output = outputs.reconstruction.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.moveaxis(output, source=0, destination=-1)
    output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8
    return Image.fromarray(output)


image_path = "360_ldm3d_rgb2.jpg"
upsample_image = enhance(image_path)

from PIL import Image
im = Image.fromarray(upsample_image)
im.save("swin2SR_2xupsample_"+image_path)



import cv2

depth_path = "360_ldm3d_depth2.png"
img = cv2.imread(depth_path)
print('Original Dimensions : ',img.shape)
scale_percent = 60 # percent of original size
times = 2
width = img.shape[1]*times
height = img.shape[0]*times
dim = (width, height)
# resize image
resized = cv2.resize(img, dim, interpolation = cv2.INTER_LINEAR)
print('Resized Dimensions : ',resized.shape)
cv2.imwrite(f"linear_{times}xupsample_"+depth_path, resized)