

if __name__ == "__main__":
    from diffusers import (
        StableDiffusionXLControlNetPipeline,
        ControlNetModel,
        UNet2DConditionModel,
    )
    import torch

    pipe_id = "stabilityai/stable-diffusion-xl-base-1.0"
    controlnet_id = "xinsir/controlnet-canny-sdxl-1.0"
    lora_id = "stabilityai/control-lora"
    lora_filename = "control-LoRAs-rank128/control-lora-canny-rank128.safetensors"


    unet = UNet2DConditionModel.from_pretrained(pipe_id, subfolder="unet", torch_dtype=torch.float16).to("cuda")
    controlnet = ControlNetModel.from_unet(unet).to(device="cuda", dtype=torch.float16)
    controlnet.load_lora_weights(lora_id, weight_name=lora_filename, controlnet_config=controlnet.config)

    from diffusers import AutoencoderKL
    from diffusers.utils import load_image, make_image_grid
    from PIL import Image
    import numpy as np
    import cv2

    prompt = "aerial view, a futuristic research complex in a bright foggy jungle, hard lighting"
    negative_prompt = "low quality, bad quality, sketches"

    image = load_image("https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/hf-logo.png")

    controlnet_conditioning_scale = 1.0  # recommended for good generalization

    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        pipe_id,
        unet=unet,
        controlnet=controlnet,
        vae=vae,
        torch_dtype=torch.float16,
        safety_checker=None,
    ).to("cuda")

    image = np.array(image)
    image = cv2.Canny(image, 100, 200)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    image = Image.fromarray(image)

    images = pipe(
        prompt, negative_prompt=negative_prompt, image=image,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        num_images_per_prompt=4
    ).images

    final_image = [image] + images
    grid = make_image_grid(final_image, 1, 5)
    grid.save(f"hf-logo1.png")
