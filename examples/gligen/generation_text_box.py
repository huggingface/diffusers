import os

import torch
import torchvision

from diffusers import StableDiffusionGLIGENPipeline


pipe = StableDiffusionGLIGENPipeline.from_pretrained(
    "gligen/diffusers-generation-text-box", revision="fp16", torch_dtype=torch.float16
)
pipe.to("cuda")

os.makedirs("images", exist_ok=True)

prompt = "a dog and an apple"
images = pipe(
    prompt,
    num_images_per_prompt=2,
    gligen_phrases=["a dog", "an apple"],
    gligen_boxes=[
        [0.1387, 0.2051, 0.4277, 0.7090],
        [0.4980, 0.4355, 0.8516, 0.7266],
    ],
    gligen_scheduled_sampling_beta=0.3,
    output_type="numpy",
    num_inference_steps=50,
).images

images = torch.stack([torch.from_numpy(image) for image in images]).permute(0, 3, 1, 2)

torchvision.utils.save_image(images, "images/generation_text_box.jpg", nrow=2, normalize=False)
