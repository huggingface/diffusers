# PromptDiffusion Pipeline

From the project [page](https://zhendong-wang.github.io/prompt-diffusion.github.io/)

"With a prompt consisting of a task-specific example pair of images and text guidance, and a new query image, Prompt Diffusion can comprehend the desired task and generate the corresponding output image on both seen (trained) and unseen (new) task types."

For any usage questions, please refer to the [paper](https://arxiv.org/abs/2305.01115).

Prepare models by converting them from the [checkpoint](https://huggingface.co/zhendongw/prompt-diffusion)

To convert the controlnet, use cldm_v15.yaml from the [repository](https://github.com/Zhendong-Wang/Prompt-Diffusion/tree/main/models/):

```bash
python convert_original_promptdiffusion_to_diffusers.py --checkpoint_path path-to-network-step04999.ckpt --original_config_file path-to-cldm_v15.yaml --dump_path path-to-output-directory
```

To learn about how to convert the fine-tuned stable diffusion model, see the [Load different Stable Diffusion formats guide](https://huggingface.co/docs/diffusers/main/en/using-diffusers/other-formats).


```py
import torch
from diffusers import UniPCMultistepScheduler
from diffusers.utils import load_image
from promptdiffusioncontrolnet import PromptDiffusionControlNetModel
from pipeline_prompt_diffusion import PromptDiffusionPipeline


from PIL import ImageOps

image_a = ImageOps.invert(load_image("https://github.com/Zhendong-Wang/Prompt-Diffusion/blob/main/images_to_try/house_line.png?raw=true"))

image_b = load_image("https://github.com/Zhendong-Wang/Prompt-Diffusion/blob/main/images_to_try/house.png?raw=true")
query = ImageOps.invert(load_image("https://github.com/Zhendong-Wang/Prompt-Diffusion/blob/main/images_to_try/new_01.png?raw=true"))

# load prompt diffusion controlnet and prompt diffusion

controlnet = PromptDiffusionControlNetModel.from_pretrained("iczaw/prompt-diffusion-diffusers", subfolder="controlnet", torch_dtype=torch.float16)
model_id = "path-to-model"
pipe = PromptDiffusionPipeline.from_pretrained("iczaw/prompt-diffusion-diffusers", subfolder="base", controlnet=controlnet, torch_dtype=torch.float16, variant="fp16")

# speed up diffusion process with faster scheduler and memory optimization
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
# remove following line if xformers is not installed
pipe.enable_xformers_memory_efficient_attention()
pipe.enable_model_cpu_offload()
# generate image
generator = torch.manual_seed(0)
image = pipe("a tortoise", num_inference_steps=20, generator=generator, image_pair=[image_a,image_b], image=query).images[0]

```
