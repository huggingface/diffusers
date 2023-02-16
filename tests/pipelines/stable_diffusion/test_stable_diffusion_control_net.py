import pytest
import torch

from diffusers import StableDiffusionControlNetPipeline


################################################################################
# PoC version
################################################################################

model_id_sd15_canny = "takuma104/control_sd15_canny"  # currntry this is private model


@pytest.mark.skip
def test_from_pretrained():
    pipe = StableDiffusionControlNetPipeline.from_pretrained(model_id_sd15_canny)
    print(pipe)


def test_from_pretrained_and_inference():
    pipe = StableDiffusionControlNetPipeline.from_pretrained(model_id_sd15_canny, torch_dtype=torch.bfloat16).to(
        "cuda"
    )
    image = pipe(prompt="an apple", num_inference_steps=15).images[0]
    image.save("/tmp/an_apple_generated.png")
    print(image.size)
