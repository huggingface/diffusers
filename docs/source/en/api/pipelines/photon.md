<!-- Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License. -->

# PhotonPipeline


Photon is a text-to-image diffusion model using simplified MMDIT architecture with flow matching for efficient high-quality image generation. The model uses T5Gemma as the text encoder and supports either Flux VAE (AutoencoderKL) or DC-AE (AutoencoderDC) for latent compression.

Key features:

- **Simplified MMDIT architecture**: Uses a simplified MMDIT architecture for image generation where text tokens are not updated through the transformer blocks
- **Flow Matching**: Employs flow matching with discrete scheduling for efficient sampling
- **Flexible VAE Support**: Compatible with both Flux VAE (8x compression, 16 latent channels) and DC-AE (32x compression, 32 latent channels)
- **T5Gemma Text Encoder**: Uses Google's T5Gemma-2B-2B-UL2 model for text encoding offering multiple language support
- **Efficient Architecture**: ~1.3B parameters in the transformer, enabling fast inference while maintaining quality

## Available models:
We offer a range of **Photon models** featuring different **VAE configurations**, each optimized for generating images at various resolutions.  
Both **fine-tuned** and **non-fine-tuned** versions are available:

- **Non-fine-tuned models** perform best with **highly detailed prompts**, capturing fine nuances and complex compositions.  
- **Fine-tuned models**, trained on the [Alchemist dataset](https://huggingface.co/datasets/yandex/alchemist), enhance the **aesthetic quality** of the base models—especially when prompts are **less detailed**.


| Model | Resolution | Fine-tuned | Distilled | Description | Suggested prompts | Suggested parameters | Recommended dtype |
|:-----:|:-----------------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|
| [`Photoroom/photon-256-t2i`](https://huggingface.co/Photoroom/photon-256-t2i)| 256 | No | No | Base model pre-trained at 256 with Flux VAE|Works best with detailed prompts in natural language|28 steps, cfg=5.0| `torch.bfloat16` |
| [`Photoroom/photon-256-t2i-sft`](https://huggingface.co/Photoroom/photon-256-t2i-sft)| 512 | Yes | No | Fine-tuned on the [Alchemist dataset](https://huggingface.co/datasets/yandex/alchemist) dataset with Flux VAE | Can handle less detailed prompts|28 steps, cfg=5.0| `torch.bfloat16` |
| [`Photoroom/photon-512-t2i`](https://huggingface.co/Photoroom/photon-512-t2i)| 512 | No | No | Base model pre-trained at 512 with Flux VAE |Works best with detailed prompts in natural language|28 steps, cfg=5.0| `torch.bfloat16` |
| [`Photoroom/photon-512-t2i-sft`](hhttps://huggingface.co/Photoroom/photon-512-t2i-sft)| 512 | Yes | No | Fine-tuned on the [Alchemist dataset](https://huggingface.co/datasets/yandex/alchemist) dataset with Flux VAE | Can handle less detailed prompts in natural language|28 steps, cfg=5.0| `torch.bfloat16` |
| [`Photoroom/photon-512-t2i-sft-distilled`](https://huggingface.co/Photoroom/photon-512-t2i-sft-distilled)| 512 | Yes | Yes | 8-step distilled model from [`Photoroom/photon-512-t2i-sft`](https://huggingface.co/Photoroom/photon-512-t2i-sft) | Can handle less detailed prompts in natural language|8 steps, cfg=1.0| `torch.bfloat16` |
| [`Photoroom/photon-512-t2i-dc-ae`](https://huggingface.co/Photoroom/photon-512-t2i-dc-ae)| 512 | No | No | Base model pre-trained at 512 with [Deep Compression Autoencoder (DC-AE)](https://hanlab.mit.edu/projects/dc-ae)|Works best with detailed prompts in natural language|28 steps, cfg=5.0| `torch.bfloat16` |
| [`Photoroom/photon-512-t2i-dc-ae-sft`](https://huggingface.co/Photoroom/photon-512-t2i-dc-ae-sft)| 512 | Yes | No | Fine-tuned on the [Alchemist dataset](https://huggingface.co/datasets/yandex/alchemist) dataset with [Deep Compression Autoencoder (DC-AE)](https://hanlab.mit.edu/projects/dc-ae) | Can handle less detailed prompts in natural language|28 steps, cfg=5.0| `torch.bfloat16` |
| [`Photoroom/photon-512-t2i-dc-ae-sft-distilled`](https://huggingface.co/Photoroom/photon-512-t2i-dc-ae-sft-distilled)| 512 | Yes | Yes | 8-step distilled model from [`Photoroom/photon-512-t2i-dc-ae-sft-distilled`](https://huggingface.co/Photoroom/photon-512-t2i-dc-ae-sft-distilled) | Can handle less detailed prompts in natural language|8 steps, cfg=1.0| `torch.bfloat16` |s

Refer to [this](https://huggingface.co/collections/Photoroom/photon-models-68e66254c202ebfab99ad38e) collection for more information.

## Loading the Pipeline

```py
from diffusers.pipelines.photon import PhotonPipeline

# Load pipeline - VAE and text encoder will be loaded from HuggingFace
pipe = PhotonPipeline.from_pretrained("Photoroom/photon-512-t2i-sft", torch_dtype=torch.bfloat16)
pipe.to("cuda")

prompt = "A front-facing portrait of a lion the golden savanna at sunset."
image = pipe(prompt, num_inference_steps=28, guidance_scale=5.0).images[0]
image.save("photon_output.png")
```

### Manual Component Loading

You can also load components individually:

```py
import torch
from diffusers import PhotonPipeline
from diffusers.models import AutoencoderKL, AutoencoderDC
from diffusers.models.transformers.transformer_photon import PhotonTransformer2DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from transformers import T5GemmaModel, GemmaTokenizerFast

# Load transformer
transformer = PhotonTransformer2DModel.from_pretrained(
    "Photoroom/photon-512-t2i-sft", subfolder="transformer"
).to(dtype=torch.bfloat16)

# Load scheduler
scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
    "Photoroom/photon-512-t2i-sft", subfolder="scheduler"
)

# Load T5Gemma text encoder
t5gemma_model = T5GemmaModel.from_pretrained("google/t5gemma-2b-2b-ul2")
text_encoder = t5gemma_model.encoder.to(dtype=torch.bfloat16)
tokenizer = GemmaTokenizerFast.from_pretrained("google/t5gemma-2b-2b-ul2")
tokenizer.model_max_length = 256
# Load VAE - choose either Flux VAE or DC-AE
# Flux VAE (16 latent channels):
vae = AutoencoderKL.from_pretrained("black-forest-labs/FLUX.1-dev", subfolder="vae").to(dtype=torch.bfloat16)
# Or DC-AE (32 latent channels):
# vae = AutoencoderDC.from_pretrained("mit-han-lab/dc-ae-f32c32-sana-1.0-diffusers")

pipe = PhotonPipeline(
    transformer=transformer,
    scheduler=scheduler,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    vae=vae
)
pipe.to("cuda")
```

## VAE Variants

Photon supports two VAE configurations:

### Flux VAE (AutoencoderKL)
- **Compression**: 8x spatial compression
- **Latent channels**: 16
- **Model**: `black-forest-labs/FLUX.1-dev` (subfolder: "vae")
- **Use case**: Balanced quality and speed

### DC-AE (AutoencoderDC)
- **Compression**: 32x spatial compression
- **Latent channels**: 32
- **Model**: `mit-han-lab/dc-ae-f32c32-sana-1.0-diffusers`
- **Use case**: Higher compression for faster processing

The VAE type is automatically determined from the checkpoint's `model_index.json` configuration.

## Generation Parameters

Key parameters for image generation:

- **num_inference_steps**: Number of denoising steps (default: 28). More steps generally improve quality at the cost of speed.
- **guidance_scale**: Classifier-free guidance strength (default: 4.0). Higher values produce images more closely aligned with the prompt.
- **height/width**: Output image dimensions (default: 512x512). Can be customized in the checkpoint configuration.

```py
# Example with custom parameters
import torch
from diffusers.pipelines.photon import PhotonPipeline
pipe = PhotonPipeline.from_pretrained("Photoroom/photon-512-t2i-sft", torch_dtype=torch.bfloat16)
pipe = pipe(
    prompt = "A front-facing portrait of a lion the golden savanna at sunset."
    num_inference_steps=28,
    guidance_scale=4.0,
    height=512,
    width=512,
    generator=torch.Generator("cuda").manual_seed(42)
).images[0]
```

## Memory Optimization

For memory-constrained environments:

```py
import torch
from diffusers.pipelines.photon import PhotonPipeline

pipe = PhotonPipeline.from_pretrained("Photoroom/photon-512-t2i-sft", torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()  # Offload components to CPU when not in use

# Or use sequential CPU offload for even lower memory
pipe.enable_sequential_cpu_offload()
```

## PhotonPipeline

[[autodoc]] PhotonPipeline
  - all
  - __call__

## PhotonPipelineOutput

[[autodoc]] pipelines.photon.pipeline_output.PhotonPipelineOutput
