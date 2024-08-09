# AnyTextPipeline Pipeline

From the project [page](https://zhendong-wang.github.io/prompt-diffusion.github.io/)

"With a prompt consisting of a task-specific example pair of images and text guidance, and a new query image, Prompt Diffusion can comprehend the desired task and generate the corresponding output image on both seen (trained) and unseen (new) task types."

For any usage questions, please refer to the [paper](https://arxiv.org/abs/2305.01115).

Prepare models by converting them from the [checkpoint](https://huggingface.co/zhendongw/prompt-diffusion)

To convert the controlnet, use cldm_v15.yaml from the [repository](https://github.com/Zhendong-Wang/Prompt-Diffusion/tree/main/models/):

```sh
python convert_original_anytext_to_diffusers.py --checkpoint_path path-to-network-step04999.ckpt --original_config_file path-to-cldm_v15.yaml --dump_path path-to-output-directory
```

To learn about how to convert the fine-tuned stable diffusion model, see the [Load different Stable Diffusion formats guide](https://huggingface.co/docs/diffusers/main/en/using-diffusers/other-formats).


```py
import torch
from pipeline_anytext import AnyTextPipeline
from text_controlnet import TextControlNetModel
from diffusers import DDIMScheduler


controlnet = TextControlNetModel.from_pretrained("a/b", subfolder="controlnet", torch_dtype=torch.float16)
model_id = "path-to-model"
pipe = AnyTextPipeline.from_pretrained("a/b", subfolder="base", controlnet=controlnet, torch_dtype=torch.float16, variant="fp16")

# speed up diffusion process with faster scheduler and memory optimization
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
# uncomment following line if torch<2.0
#pipe.enable_xformers_memory_efficient_attention()
pipe.enable_model_cpu_offload()
# generate image
generator = torch.Generator("cpu").manual_seed(0)
image = pipe("photo of caramel macchiato coffee on the table, top-down perspective, with "Any" "Text" written on it using cream", num_inference_steps=20, generator=generator).images[0]
image
```
