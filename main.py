import os
os.environ["DIFFUSERS_ENABLE_HUB_KERNELS"] = "yes"

# Debug: Verify the env var is set
print(f"DIFFUSERS_ENABLE_HUB_KERNELS = {os.environ.get('DIFFUSERS_ENABLE_HUB_KERNELS')}")

import torch
from diffusers import FluxPipeline
from diffusers.quantizers import PipelineQuantizationConfig

# Debug: Check if diffusers sees the env var
from diffusers.models.attention_dispatch import DIFFUSERS_ENABLE_HUB_KERNELS
print(f"Diffusers sees DIFFUSERS_ENABLE_HUB_KERNELS = {DIFFUSERS_ENABLE_HUB_KERNELS}")

# ✅ 3. Load pipeline with quantization
model_id = "black-forest-labs/FLUX.1-dev"
pipe = FluxPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    quantization_config=PipelineQuantizationConfig(
        quant_backend="bitsandbytes_4bit",
        quant_kwargs={
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": torch.bfloat16,
        },
        components_to_quantize=["transformer"],
    ),
).to("cuda")

pipe.transformer.set_attention_backend("_flash_hub")

prompt = "A cat holding a sign that says 'hello world'"
image = pipe(prompt, num_inference_steps=28, guidance_scale=4.0).images[0]
image.save("output.png")