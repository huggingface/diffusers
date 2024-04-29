# Stable Diffusion

## Overview

ELLA was proposed in [ELLA: Equip Diffusion Models with LLM for Enhanced Semantic Alignment](https://arxiv.org/pdf/2403.05135) by Xiwei Hu, Rui Wang, Yixiao Fang, Bin Fu, Pei Cheng, and Gang Yu

The summary of the model is the following:
*Diffusion models have demonstrated remarkable performancein the domain of text-to-image generation. However, most widely usedmodels still employ CLIP as their text encoder, which constrains theirability to comprehend dense prompts, encompassing multiple objects,detailed attributes, complex relationships, long-text alignment, etc. Inthis paper, we introduce anEfficientLargeLanguage Model Adapter,termed ELLA, which equips text-to-image diffusion models with powerful Large Language Models (LLM) to enhance text alignment without training of either U-Net or LLM. To seamlessly bridge two pre-trained models, we investigate a range of semantic alignment connector designs and propose a novel module, the Timestep-Aware Semantic Connector (TSC), which dynamically extracts timestep-dependent conditions from LLM. Our approach adapts semantic features at different stages of the denoising process, assisting diffusion models in interpreting lengthy and intricate prompts over sampling timesteps. Additionally, ELLA can be readily incorporated with community models and tools to improve their prompt-following capabilities. To assess text-to-image models in dense prompt following, we introduce Dense Prompt Graph Benchmark (DPGBench), a challenging benchmark consisting of 1K dense prompts. Extensive experiments demonstrate the superiority of ELLA in dense prompt following compared to state-of-the-art methods, particularly in multiple object compositions involving diverse attributes and relationships.

## Examples:

### Impoting all the required pipelines
```python
from diffusers_plus_plus import EllaDiffusionPipeline, ELLA, DPMSolverMultistepScheduler
```

### Load pretrained ELLA weights from the hub provided by the authors of the paper
```python
ELLA = ELLA.from_pretrained('shauray/ELLA_SD15')
```

### Load all the parts of the pipeline namely the scheduler, unet, vae etc. and this can be used with adapters like T2I and IP-Adapter
```python
ella_pipeline = EllaDiffusionPipeline.from_pretrained("Justin-Choo/epiCRealism-Natural_Sin_RC1_VAE",ELLA=ELLA, requires_safety_checker=False)
ella_pipeline.scheduler = DPMSolverMultistepScheduler.from_config(ella_pipeline.scheduler.config)
ella_pipeline = ella_pipeline.to("cuda")
```

### provide a prompt which would then be converted into llm token outputs in order to feed it through ELLA
```python
prompt = "a beautiful portrait of an empress in her garden"
negative_prompt = ""
```

### Generate and save the image
```python
image = ella_pipeline(prompt, negative_prompt=negative_prompt, guidance=7,num_inference_steps=30, height=768, width=512).images[0]

image.save("black_to_blue.png")
```

### Inference Example
|  ELLA NOT-Fixed Embedding Length | ELLA Fixed Embedding Length | SD15  |
| ----------- | ----------- | ----------- |
| ![Example Image](https://drive.google.com/uc?id=1zgFb3ELhftBem2PTmZVhhbBahxQBSYX0) | ![Example Image](https://drive.google.com/uc?id=1m4vjEnguRWM8ZTGdXTA25A4xZeoKuyhh) |  ![Example Image](https://drive.google.com/uc?id=1Te5V1Htku-3zZyiFS1ws4LL15zfhlvDh) |
