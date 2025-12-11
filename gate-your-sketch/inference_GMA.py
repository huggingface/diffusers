import torch
from diffusers import (
    StableDiffusionXLAdapterPipeline,
    AutoencoderKL,
    EulerAncestralDiscreteScheduler,
    GatedMultiAdapter
)
from diffusers.utils import load_image
from diffusers import StableDiffusionXLAdapterPipeline, T2IAdapter, EulerAncestralDiscreteScheduler, AutoencoderKL,MultiAdapter,GatedMultiAdapter
from diffusers.utils import load_image, make_image_grid
from controlnet_aux.midas import MidasDetector
import torch
from controlnet_aux.canny import CannyDetector

device = "cuda"

# -----------------------------
# 1. Load trained GatedMultiAdapter
# -----------------------------
adapters = GatedMultiAdapter.from_pretrained(
    "/home/ubuntu/gate-your-sketch-training_output/sdxl_GMA_withFiLM_t1.0_res512_lr1e-7_bs1x4_seed42_step2000/t2iadapter",
    torch_dtype=torch.float16,
)

# load euler_a scheduler
model_id = 'stabilityai/stable-diffusion-xl-base-1.0'
euler_a = EulerAncestralDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
vae=AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
model_id, vae=vae, adapter=adapters, scheduler=euler_a, torch_dtype=torch.float16, variant="fp16",
).to(device)
# pipe.enable_xformers_memory_efficient_attention()

midas_depth = MidasDetector.from_pretrained(
"valhalla/t2iadapter-aux-models", filename="dpt_large_384.pt", model_type="dpt_large"
).to("cuda")
canny_detector = CannyDetector()
url = "https://huggingface.co/Adapter/t2iadapter/resolve/main/figs_SDXLV1.0/org_mid.jpg"
image = load_image(url)
depth_midas_control_image = midas_depth(
image, detect_resolution=512, image_resolution=1024
)
canny_control_image = canny_detector(image, detect_resolution=384, image_resolution=1024)#.resize((1024, 1024))
prompt = "A photo of a room, 4k photo, highly detailed"
negative_prompt = "anime, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured"
control_weights=[0.6,0.6]

gen_images = pipe(
prompt=prompt,
negative_prompt=negative_prompt,
image=[canny_control_image, depth_midas_control_image],
num_inference_steps=30,
adapter_conditioning_scale=control_weights,
guidance_scale=7.5,
).images[0]
gen_images.save('out_GMA_ckpt_1000_nonsep_timeemb_film.png')