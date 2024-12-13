### UFOGen Scheduler

[UFOGen](https://arxiv.org/abs/2311.09257) is a generative model designed for fast one-step text-to-image generation, trained via adversarial training starting from an initial pretrained diffusion model such as Stable Diffusion. `scheduling_ufogen.py` implements a onestep and multistep sampling algorithm for UFOGen models compatible with pipelines like `StableDiffusionPipeline`. A usage example is as follows:

```py
import torch
from diffusers import StableDiffusionPipeline

from scheduling_ufogen import UFOGenScheduler

# NOTE: currently, I am not aware of any publicly available UFOGen model checkpoints trained from SD v1.5.
ufogen_model_id_or_path = "/path/to/ufogen/model"
pipe = StableDiffusionPipeline(
    ufogen_model_id_or_path,
    torch_dtype=torch.float16,
)

# You can initialize a UFOGenScheduler as follows:
pipe.scheduler = UFOGenScheduler.from_config(pipe.scheduler.config)

prompt = "Three cats having dinner at a table at new years eve, cinematic shot, 8k."

# Onestep sampling
onestep_image = pipe(prompt, num_inference_steps=1).images[0]

# Multistep sampling
multistep_image = pipe(prompt, num_inference_steps=4).images[0]
```