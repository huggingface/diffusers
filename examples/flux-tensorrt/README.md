# 🚀 Flux TensorRT Pipelines

This project provides **TensorRT-accelerated pipelines** for Flux models, enabling **faster inference** with static and dynamic shapes.

## ✅ Supported Pipelines
- ✅ `FluxPipeline` (Supported)
- ⏳ `FluxImg2ImgPipeline` (Coming soon)
- ⏳ `FluxInpaintPipeline` (Coming soon)
- ⏳ `FluxFillPipeline` (Coming soon)
- ⏳ `FluxKontextPipeline` (Coming soon)
- ⏳ `FluxKontextInpaintPipeline` (Coming soon)

---


## Installation
```bash
cd diffusers/examples/flux-tensorrt
pip install -r requirements.txt
```

## ⚙️ Build Flux with TensorRT

Before building, make sure you have the ONNX checkpoints ready.
You can either download the official [Flux ONNX](https://huggingface.co/black-forest-labs/FLUX.1-dev-onnx) checkpoints from Hugging Face, or export your own.

```bash
huggingface-cli download black-forest-labs/FLUX.1-dev-onnx --local-dir onnx
```

Build each component individually. For example, to build the **Transformer engine**:
```python
from module.transformers import FluxTransformerModel

engine_path = "checkpoints_trt/transformer/engine.plan"
engine_transformer = FluxTransformerModel(engine_path=engine_path,build=True)

# Build tranformer engine
transformer_input_profile = engine_transformer.get_input_profile(
    opt_batch_size=1,
    opt_image_height=1024,
    opt_image_width=1024,
    static_batch = True,
    dynamic_shape= True
)
engine_transformer.build(
    onnx_path="onnx/transformer.opt/bf16/model.onnx", #Replace your onnx path
    input_profile=transformer_input_profile,
)
```

You can convert all ONNX checkpoints to TensorRT engines with a single command:
```bash
python convert_trt.py
```

## 🖼️ Inference with Flux TensorRT
Example: Full Pipeline with All Engines

```python
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
```

Example: Transformer Only (Other Modules on Torch)
```python
pipeline = FluxPipelineTRT.from_pretrained(
   "black-forest-labs/FLUX.1-dev", 
   torch_dtype=torch.bfloat16, 
   engine_transformer=engine_transformer,
)
pipeline.to("cuda")
```

## 📌 Notes

- Ensure correct CUDA / TensorRT versions are installed.

- Always match the `.plan` engine files with the TensorRT version used for building.

- For best performance, prefer static shapes unless dynamic batching is required.