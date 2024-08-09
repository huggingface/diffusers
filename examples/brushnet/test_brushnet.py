from diffusers import StableDiffusionBrushNetPipeline, BrushNetModel, UniPCMultistepScheduler
import torch
import cv2
import numpy as np
from PIL import Image

# choose the base model here
base_model_path = "data/ckpt/realisticVisionV60B1_v51VAE"
# base_model_path = "runwayml/stable-diffusion-v1-5"

# input brushnet ckpt path
brushnet_path = "data/ckpt/segmentation_mask_brushnet_ckpt"

# choose whether using blended operation
blended = False

# input source image / mask image path and the text prompt
image_path="examples/brushnet/src/test_image.jpg"
mask_path="examples/brushnet/src/test_mask.jpg"
caption="A cake on the table."

# conditioning scale
brushnet_conditioning_scale=1.0

brushnet = BrushNetModel.from_pretrained(brushnet_path, torch_dtype=torch.float16)
pipe = StableDiffusionBrushNetPipeline.from_pretrained(
    base_model_path, brushnet=brushnet, torch_dtype=torch.float16, low_cpu_mem_usage=False
)

# speed up diffusion process with faster scheduler and memory optimization
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
# remove following line if xformers is not installed or when using Torch 2.0.
# pipe.enable_xformers_memory_efficient_attention()
# memory optimization.
pipe.enable_model_cpu_offload()

init_image = cv2.imread(image_path)[:,:,::-1]
mask_image = 1.*(cv2.imread(mask_path).sum(-1)>255)[:,:,np.newaxis]
init_image = init_image * (1-mask_image)

init_image = Image.fromarray(init_image.astype(np.uint8)).convert("RGB")
mask_image = Image.fromarray(mask_image.astype(np.uint8).repeat(3,-1)*255).convert("RGB")

generator = torch.Generator("cuda").manual_seed(1234)

image = pipe(
    caption, 
    init_image, 
    mask_image, 
    num_inference_steps=50, 
    generator=generator,
    brushnet_conditioning_scale=brushnet_conditioning_scale
).images[0]

if blended:
    image_np=np.array(image)
    init_image_np=cv2.imread(image_path)[:,:,::-1]
    mask_np = 1.*(cv2.imread(mask_path).sum(-1)>255)[:,:,np.newaxis]

    # blur, you can adjust the parameters for better performance
    mask_blurred = cv2.GaussianBlur(mask_np*255, (21, 21), 0)/255
    mask_blurred = mask_blurred[:,:,np.newaxis]
    mask_np = 1-(1-mask_np) * (1-mask_blurred)

    image_pasted=init_image_np * (1-mask_np) + image_np*mask_np
    image_pasted=image_pasted.astype(image_np.dtype)
    image=Image.fromarray(image_pasted)

image.save("output.png")