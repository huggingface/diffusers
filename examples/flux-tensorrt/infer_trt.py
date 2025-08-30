from pipeline_flux_trt import FluxPipelineTRT
from cuda import cudart
import torch

from module.transformers import FluxTransformerModel 
from module.vae import VAEModel
from module.t5xxl import T5XXLModel
from module.clip import CLIPModel
import time

# Local path for each engine
engine_transformer_path = "checkpoints_trt/transformer/engine.plan"
engine_vae_path = "checkpoints_trt/vae/engine.plan"
engine_t5xxl_path = "checkpoints_trt/t5/engine.plan"
engine_clip_path = "checkpoints_trt/clip/engine.plan"

# Create stream for each engine
stream = cudart.cudaStreamCreate()[1]

# Create engine for each model
engine_transformer = FluxTransformerModel(engine_transformer_path, stream)
engine_vae = VAEModel(engine_vae_path, stream)
engine_t5xxl = T5XXLModel(engine_t5xxl_path, stream)
engine_clip = CLIPModel(engine_clip_path, stream)

# Create pipeline
pipeline = FluxPipelineTRT.from_pretrained(
            "black-forest-labs/FLUX.1-dev", 
            torch_dtype=torch.bfloat16, 
            engine_transformer=engine_transformer,
            engine_vae=engine_vae,
            engine_text_encoder=engine_clip,
            engine_text_encoder_2= engine_t5xxl,
            )
pipeline.to("cuda")


prompt = "A cat holding a sign that says hello world"
generator = torch.Generator(device="cuda").manual_seed(42)
image = pipeline(prompt, num_inference_steps=28, guidance_scale=3.0, generator=generator).images[0]


image.save("test_pipeline.png")