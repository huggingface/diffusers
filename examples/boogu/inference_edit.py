import torch
from PIL import Image

from diffusers.pipelines.boogu import BooguImagePipeline


MODEL_PATH = "Boogu/Boogu-Image-0.1-Edit"

# Negative prompt steering quality away from common artifacts. With text_guidance_scale > 1
# the model guides away from this prompt, so it noticeably improves style adherence.
NEGATIVE_INSTRUCTION = (
    "(((deformed))), blurry, over saturation, bad anatomy, disfigured, poorly drawn face, "
    "mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), fused fingers, messy drawing, "
    "broken legs censor, censored, censor_bar"
)

pipe = BooguImagePipeline.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16)
pipe = pipe.to("cuda")

images = pipe(
    instruction="把图片风格调整为彩铅插画。",
    negative_instruction=NEGATIVE_INSTRUCTION,
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
