import os
import torch
import torchvision
from diffusers import StableDiffusionGLIGENPipeline

from PIL import Image

pipe = StableDiffusionGLIGENPipeline.from_pretrained(
    "gligen/diffusers-inpainting-text-box", revision="fp16", torch_dtype=torch.float16
)
pipe.to("cuda")

os.makedirs("images", exist_ok=True)

prompt = "a birthday cake"

images = pipe(
    prompt,
    num_images_per_prompt=2,
    gligen_phrases=["a birthday cake"],
    gligen_inpaint_image=Image.open("resources/livingroom_marble.jpg").convert("RGB"),
    gligen_boxes=[
        [0.5243, 0.7027, 0.7136, 0.8341],
    ],
    gligen_scheduled_sampling_beta=1,
    output_type="numpy",
    num_inference_steps=50,
).images

images = torch.stack([torch.from_numpy(image) for image in images]).permute(0, 3, 1, 2)

torchvision.utils.save_image(images, "images/inpaint_text_box.jpg", nrow=2, normalize=False)
