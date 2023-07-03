import os
import numpy as np
import torch
from PIL import Image
from diffusers import WuerstchenPriorPipeline, WuerstchenGeneratorPipeline


def numpy_to_pil(images: np.ndarray) -> list[Image]:
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


prior_pipeline = WuerstchenPriorPipeline.from_pretrained("C:\\Users\\d6582\\Documents\\ml\\diffusers\\scripts\\kashif\\WuerstchenPriorPipeline", torch_dtype=torch.float16)
generator_pipeline = WuerstchenGeneratorPipeline.from_pretrained("C:\\Users\\d6582\\Documents\\ml\\diffusers\\scripts\\kashif\\WuerstchenGeneratorPipeline", torch_dtype=torch.float16)
prior_pipeline = prior_pipeline.to("cuda")
generator_pipeline = generator_pipeline.to("cuda")

negative_prompt = "low resolution, low detail, bad quality, blurry"
# negative_prompt = ""
# caption = "Bee flying out of a glass jar in a green and red leafy basket, glass and lens flare, diffuse lighting elegant"
# caption = "princess | centered| key visual| intricate| highly detailed| breathtaking beauty| precise lineart| vibrant| comprehensive cinematic| Carne Griffiths| Conrad Roset"
caption = input("Prompt please: ")
while caption != "q":
    prior_output = prior_pipeline(caption, num_images_per_prompt=4, negative_prompt=negative_prompt)
    generator_output = generator_pipeline(prior_output.image_embeds, prior_output.text_embeds, output_type="np").images
    images = numpy_to_pil(generator_output)

    os.makedirs("samples", exist_ok=True)
    for i, image in enumerate(images):
        image.save(os.path.join("samples", caption.replace(" ", "_").replace("|", "") + f"_{i}.png"))

    caption = input("Prompt please: ")

