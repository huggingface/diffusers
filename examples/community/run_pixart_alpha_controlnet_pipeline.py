import sys
import os
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from pipeline_pixart_alpha_controlnet import PixArtAlphaControlnetPipeline
from diffusers.utils import load_image

from diffusers.image_processor import PixArtImageProcessor

from controlnet_aux import HEDdetector

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pixart.controlnet_pixart_alpha import PixArtControlNetAdapterModel

controlnet_repo_id = "raulc0399/pixart-alpha-hed-controlnet"

weight_dtype = torch.float16
image_size = 1024

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(0)

# load controlnet
controlnet = PixArtControlNetAdapterModel.from_pretrained(
    controlnet_repo_id,
    torch_dtype=weight_dtype,
    use_safetensors=True,
).to(device)

pipe = PixArtAlphaControlnetPipeline.from_pretrained(
    "PixArt-alpha/PixArt-XL-2-1024-MS",
    controlnet=controlnet,
    torch_dtype=weight_dtype,
    use_safetensors=True,
).to(device)

images_path = "images"
control_image_file = "0_7.jpg"

# prompt = "cinematic photo of superman in action . 35mm photograph, film, bokeh, professional, 4k, highly detailed"
# prompt = "yellow modern car, city in background, beautiful rainy day"
# prompt = "modern villa, clear sky, suny day . 35mm photograph, film, bokeh, professional, 4k, highly detailed"
# prompt = "robot dog toy in park . 35mm photograph, film, bokeh, professional, 4k, highly detailed"
# prompt = "purple car, on highway, beautiful sunny day"
# prompt = "realistical photo of a loving couple standing in the open kitchen of the living room, cooking ."
prompt = "battleship in space, galaxy in background"

control_image_name = control_image_file.split('.')[0]

control_image = load_image(f"{images_path}/{control_image_file}")
print(control_image.size)
height, width = control_image.size

hed = HEDdetector.from_pretrained("lllyasviel/Annotators")

condition_transform = T.Compose([
    T.Lambda(lambda img: img.convert('RGB')),
    T.CenterCrop([image_size, image_size]),
])

control_image = condition_transform(control_image)
hed_edge = hed(control_image, detect_resolution=image_size, image_resolution=image_size)

hed_edge.save(f"{images_path}/{control_image_name}_hed.jpg")

# run pipeline
with torch.no_grad():
    out = pipe(
        prompt=prompt,
        image=hed_edge,
        num_inference_steps=14,
        guidance_scale=4.5,
        height=image_size,
        width=image_size,
    )

    out.images[0].save(f"{images_path}//{control_image_name}_output.jpg")
    