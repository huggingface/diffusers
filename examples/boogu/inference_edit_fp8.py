import os

import torch
from PIL import Image

from diffusers import BooguImageTransformer2DModel
from diffusers.pipelines.boogu import BooguImagePipeline


def _disable_deepgemm_for_fp8_vlm() -> None:
    # For transformers >= 5.11.0
    os.environ["TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR"] = "1"

    try:
        import transformers.integrations.finegrained_fp8 as fg_fp8
    except Exception:
        return

    def _raise_import_error(*args, **kwargs):
        raise ImportError("DeepGEMM disabled; forcing Triton finegrained-fp8 fallback.")

    if hasattr(fg_fp8, "deepgemm_fp8_fp4_linear"):
        # For 5.10.1 <= transformers < 5.11.0
        fg_fp8.deepgemm_fp8_fp4_linear = _raise_import_error
    elif hasattr(fg_fp8, "_load_deepgemm_kernel"):
        # For 5.5.0 <= transoformers < 5.10.1
        fg_fp8._load_deepgemm_kernel = _raise_import_error


_disable_deepgemm_for_fp8_vlm()

MODEL_PATH = "Boogu/Boogu-Image-0.1-Edit-fp8"

# Negative prompt steering quality away from common artifacts. With text_guidance_scale > 1
# the model guides away from this prompt, so it noticeably improves style adherence.
NEGATIVE_INSTRUCTION = (
    "(((deformed))), blurry, over saturation, bad anatomy, disfigured, poorly drawn face, "
    "mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), fused fingers, messy drawing, "
    "broken legs censor, censored, censor_bar"
)

if not os.path.exists("base.png"):
    raise FileNotFoundError("base.png not found — run inference_base.py first to generate the reference image.")

transformer = BooguImageTransformer2DModel.from_pretrained(
    MODEL_PATH,
    subfolder="transformer",
    torch_dtype=torch.bfloat16,
    use_safetensors=False,
)
pipe = BooguImagePipeline.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, transformer=transformer)
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
images[0].save("edit_fp8.png")
print("Inference OK, saved edit_fp8.png")
