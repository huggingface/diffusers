### PromptDiffusion Pipeline

From the project [page](https://zhendong-wang.github.io/prompt-diffusion.github.io/)

"With a prompt consisting of a task-specific example pair of images and text guidance, and a new query image, Prompt Diffusion can comprehend the desired task and generate the corresponding output image on both seen (trained) and unseen (new) task types."

For any usage questions, please refer to the [paper](https://arxiv.org/abs/2305.01115).


```py
# Prepare models by converting them from the checkpoint https://huggingface.co/zhendongw/prompt-diffusion
import torch
import diffusers
from diffusers import UniPCMultistepScheduler
from diffusers.utils import load_image
from promptdiffusioncontrolnet import PromptDiffusionControlNetModel
from diffusers.pipelines.pipeline_utils import DiffusionPipeline


from PIL import Image,ImageOps

image_a = ImageOps.invert(load_image("https://github.com/Zhendong-Wang/Prompt-Diffusion/blob/main/images_to_try/house_line.png?raw=true"))

image_b = load_image("https://github.com/Zhendong-Wang/Prompt-Diffusion/blob/main/images_to_try/house.png?raw=true")
query = ImageOps.invert(load_image("https://github.com/Zhendong-Wang/Prompt-Diffusion/blob/main/images_to_try/new_01.png?raw=true"))

# load prompt diffusion control net and prompt diffusion

controlnet = PromptDiffusionControlNetModel.from_pretrained("path-to-promptdiffusion-controlnet", torch_dtype=torch.float16)
model_id = "path-to-model"
pipe = DiffusionPipeline.from_pretrained(model_id, controlnet=controlnet, torch_dtype=torch.float16, variant="fp16", custom_pipeline="path-to-pipeline_prompt_diffusion")

# speed up diffusion process with faster scheduler and memory optimization
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
# remove following line if xformers is not installed
pipe.enable_xformers_memory_efficient_attention()
pipe.enable_model_cpu_offload()
# generate image
generator = torch.manual_seed(0)
image = pipe("a tortoise", num_inference_steps=20, generator=generator, image_pair=[image_a,image_b], image=query).images[0]

```
