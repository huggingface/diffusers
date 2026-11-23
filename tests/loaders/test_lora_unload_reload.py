from diffusers import StableDiffusionPipeline
import torch
import pytest

def test_unload_reload_same_adapter():
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float32
    ).to("cpu")

    lora_repo = "latent-consistency/lcm-lora-sdv1-5"

    # Load and activate LoRA
    pipe.load_lora_weights(lora_repo)
    adapters = pipe.get_list_adapters()
    adapter_name = list(adapters["unet"])[0]

    pipe.set_adapters([adapter_name], [1.0])

    # Unload
    pipe.unload_lora_weights()

    # Reload
    pipe.load_lora_weights(lora_repo)
    adapters = pipe.get_list_adapters()
    adapter_name = list(adapters["unet"])[0]

    pipe.set_adapters([adapter_name], [0.8])
