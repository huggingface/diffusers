import torch

from diffusers import FluxPipeline, FluxTransformer2DModel, GGUFQuantizationConfig

ckpt_path = (
    "/home/mozf/LLM/flux1-dev-Q4_0.gguf"
)
transformer = FluxTransformer2DModel.from_single_file(
    ckpt_path,
    quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
    torch_dtype=torch.bfloat16,
)
pipe = FluxPipeline.from_pretrained(
    "/home/mozf/LLM/FLUX.1-dev",
    transformer=transformer,
    torch_dtype=torch.bfloat16,
).to("cuda")


# pipe.enable_model_cpu_offload()
pipe.transformer.to(memory_format=torch.channels_last)
pipe.transformer.compile(mode="reduce-overhead", fullgraph=True)

prompt = "A cat holding a sign that says hello world"
image = pipe(prompt, generator=torch.manual_seed(0)).images[0]
# image.save("flux-gguf.png")

prompt = "A cat holding a sign that says hello world"
image = pipe(prompt, generator=torch.manual_seed(0)).images[0]
image.save("flux-gguf.png")
