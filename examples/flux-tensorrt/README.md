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

## ⚙️ Building Flux with TensorRT

We follow the official [NVIDIA/TensorRT](https://github.com/NVIDIA/TensorRT) repository to build TensorRT.

> **Note:**  
> TensorRT was originally built with `diffusers==0.31.1`.  
> Currently, we recommend using:
> - one **venv** for building, and  
> - another **venv** for inference.  

(🔜 TODO: Build scripts for the latest `diffusers` will be added later.)

### Installation
```bash
git clone https://github.com/NVIDIA/TensorRT
cd TensorRT/demo/Diffusion

pip install tensorrt-cu12==10.13.2.6
pip install -r requirements.txt
```

### ⚡ Fast Building with Static Shapes
```bash
# BF16
python3 demo_txt2img_flux.py "a beautiful photograph of Mt. Fuji during cherry blossom" --hf-token=$HF_TOKEN --bf16 --download-onnx-models

# FP8
python3 demo_txt2img_flux.py "a beautiful photograph of Mt. Fuji during cherry blossom" --hf-token=$HF_TOKEN --quantization-level 4 --fp8 --download-onnx-models

# FP4
python3 demo_txt2img_flux.py "a beautiful photograph of Mt. Fuji during cherry blossom" --hf-token=$HF_TOKEN --fp4 --download-onnx-models
```

- To build with dynamic shape, add: `--build-dynamic-shape`.
- To build with static batch, add  `--build-static-batch`.

ℹ️ For more details, run:
`python demo_txt2img_flux.py --help`

## 🖼️ Inference with Flux TensorRT
Create a new venv (or update diffusers, peft in your existing one), then run fast inference using TensorRT engines.

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
engine_transformer_path = "path/to/transformer/engine_trt10.13.2.6.plan"
engine_vae_path = "path/to/vae/engine_trt10.13.2.6.plan"
engine_t5xxl_path = "path/to/t5/engine_trt10.13.2.6.plan"
engine_clip_path = "path/to/clip/engine_trt10.13.2.6.plan"

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