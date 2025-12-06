from diffusers import StableDiffusionXLAdapterPipeline, T2IAdapter, EulerAncestralDiscreteScheduler, AutoencoderKL,MultiAdapter,GatedMultiAdapter
from diffusers.utils import load_image, make_image_grid
from controlnet_aux.midas import MidasDetector
import torch
from controlnet_aux.canny import CannyDetector

device='cuda:0'

# load adapter
depth_midas_adapter = T2IAdapter.from_pretrained(
"TencentARC/t2i-adapter-depth-midas-sdxl-1.0", torch_dtype=torch.float16, varient="fp16"
).to(device)
canny_adapter = T2IAdapter.from_pretrained("TencentARC/t2i-adapter-canny-sdxl-1.0", torch_dtype=torch.float16, varient="fp16").to(device)
adapters = GatedMultiAdapter([depth_midas_adapter,canny_adapter])

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
image=[depth_midas_control_image,canny_control_image],
num_inference_steps=30,
adapter_conditioning_scale=control_weights,
guidance_scale=7.5,
).images[0]
gen_images.save('out_mid_GMA1.png')