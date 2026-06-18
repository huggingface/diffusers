import torch
from PIL import Image

from diffusers.pipelines.boogu import BooguImagePipeline


MODEL_PATH = "Boogu/Boogu-Image-0.1-Edit"

pipe = BooguImagePipeline.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16)
pipe = pipe.to("cuda")

images = pipe(
    instruction="把图片风格调整为彩铅插画。",
    input_images=[Image.open("base.png").convert("RGB")],
    height=1024,
    width=1024,
    num_inference_steps=50,
    text_guidance_scale=4.0,
    image_guidance_scale=1.0,
).images

assert len(images) == 1
images[0].save("edit.png")
print("Inference OK, saved edit.png")
