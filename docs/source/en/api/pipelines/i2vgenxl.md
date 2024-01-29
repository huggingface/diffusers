# Image-to-Video Generation with I2VGen-XL

## Overview

[I2VGen-XL: High-Quality Image-to-Video Synthesis via Cascaded Diffusion Models](https://arxiv.org/pdf/2311.04145.pdf) by Shiwei Zhang, Jiayu Wang, Yingya Zhang, Kang Zhao, Hangjie Yuan, Zhiwu Qin, Xiang Wang, Deli Zhao, Jingren Zhou

The abstract of the paper is the following:

Video synthesis has recently made remarkable strides benefiting from the rapid development of diffusion models. However, it still encounters challenges in terms of semantic accuracy, clarity and spatio-temporal continuity. They primarily arise from the scarcity of well-aligned text-video data and the complex inherent structure of videos, making it difficult for the model to simultaneously ensure semantic and qualitative excellence. In this report, we propose a cascaded I2VGen-XL approach that enhances model performance by decoupling these two factors and ensures the alignment of the input data by utilizing static images as a form of crucial guidance. I2VGen-XL consists of two stages: i) the base stage guarantees coherent semantics and preserves content from input images by using two hierarchical encoders, and ii) the refinement stage enhances the video's details by incorporating an additional brief text and improves the resolution to 1280×720. To improve the diversity, we collect around 35 million single-shot text-video pairs and 6 billion text-image pairs to optimize the model. By this means, I2VGen-XL can simultaneously enhance the semantic accuracy, continuity of details and clarity of generated videos. Through extensive experiments, we have investigated the underlying principles of I2VGen-XL and compared it with current top methods, which can demonstrate its effectiveness on diverse data. The source code and models will be publicly available at [this https URL](https://i2vgen-xl.github.io/).


## Available Pipelines

| Pipeline | Tasks | Demo
|---|---|:---:|
| [I2VGenXLPipeline](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/i2vgen_xl/pipeline_i2vgen_xl.py) | *Image-to-Video Generation with I2VGen-XL* |

## Available checkpoints

## Usage Example

```python
import torch
from diffusers import I2VGenXLPipeline
from diffusers.utils import load_image, export_to_gif

repo_id = "diffusers/i2vgen-xl"
pipeline = I2VGenXLPipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
pipeline.enable_model_cpu_offload()

image_url = "https://github.com/ali-vilab/i2vgen-xl/blob/main/data/test_images/img_0009.png?raw=true"
image = load_image(image_url).convert("RGB")
prompt = "Papers were floating in the air on a table in the library"
negative_prompt = "Distorted, discontinuous, Ugly, blurry, low resolution, motionless, static, disfigured, disconnected limbs, Ugly faces, incomplete arms"

generator = torch.manual_seed(8888)
frames = pipeline(
    prompt=prompt,
    image=image,
    negative_prompt=negative_prompt,
    generator=generator).frames[0]
video_path = export_to_gif(frames, 'i2v.gif')
```

Here is a sample output:

<table>
    <tr>
        <td><center>
        masterpiece, bestquality, sunset.
        <br>
        <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/i2vgen-xl-example.gif"
            alt="library"
            style="width: 300px;" />
        </center></td>
    </tr>
</table>