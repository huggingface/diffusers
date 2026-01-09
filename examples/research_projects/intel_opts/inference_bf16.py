import argparse

import intel_extension_for_pytorch as ipex
import torch

from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline


parser = argparse.ArgumentParser("Stable Diffusion script with intel optimization", add_help=False)
parser.add_argument("--dpm", action="store_true", help="Enable DPMSolver or not")
parser.add_argument("--steps", default=None, type=int, help="Num inference steps")
args = parser.parse_args()


device = "cpu"
prompt = "a lovely <dicoo> in red dress and hat, in the snowly and brightly night, with many brightly buildings"

model_id = "path-to-your-trained-model"
pipe = StableDiffusionPipeline.from_pretrained(model_id)
if args.dpm:
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(device)

# to channels last
pipe.unet = pipe.unet.to(memory_format=torch.channels_last)
pipe.vae = pipe.vae.to(memory_format=torch.channels_last)
pipe.text_encoder = pipe.text_encoder.to(memory_format=torch.channels_last)
if pipe.requires_safety_checker:
    pipe.safety_checker = pipe.safety_checker.to(memory_format=torch.channels_last)

# optimize with ipex
sample = torch.randn(2, 4, 64, 64)
timestep = torch.rand(1) * 999
encoder_hidden_status = torch.randn(2, 77, 768)
input_example = (sample, timestep, encoder_hidden_status)
try:
    pipe.unet = ipex.optimize(pipe.unet.eval(), dtype=torch.bfloat16, inplace=True, sample_input=input_example)
except Exception:
    pipe.unet = ipex.optimize(pipe.unet.eval(), dtype=torch.bfloat16, inplace=True)
pipe.vae = ipex.optimize(pipe.vae.eval(), dtype=torch.bfloat16, inplace=True)
pipe.text_encoder = ipex.optimize(pipe.text_encoder.eval(), dtype=torch.bfloat16, inplace=True)
if pipe.requires_safety_checker:
    pipe.safety_checker = ipex.optimize(pipe.safety_checker.eval(), dtype=torch.bfloat16, inplace=True)

# compute
seed = 666
generator = torch.Generator(device).manual_seed(seed)
generate_kwargs = {"generator": generator}
if args.steps is not None:
    generate_kwargs["num_inference_steps"] = args.steps

with torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
    image = pipe(prompt, **generate_kwargs).images[0]

# save image
image.save("generated.png")
