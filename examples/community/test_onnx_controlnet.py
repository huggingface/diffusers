import argparse
import os
import time

from PIL import Image

from diffusers import OnnxRuntimeModel, StableDiffusionImg2ImgPipeline, UniPCMultistepScheduler
from diffusers.pipelines.controlnet.pipeline_onnx_controlnet_img2img import (
    OnnxStableDiffusionControlNetImg2ImgPipeline,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sd_model",
        type=str,
        required=True,
        help="Path to the `diffusers` checkpoint to convert (either a local directory or on the Hub).",
    )

    parser.add_argument(
        "--onnx_model_dir",
        type=str,
        required=True,
        help="Path to the ONNX directory",
    )

    parser.add_argument("--qr_img_path", type=str, required=True, help="Path to the qr code image")

    args = parser.parse_args()

    qr_image = Image.open(args.qr_img_path)
    qr_image = qr_image.resize((512, 512))

    # init stable diffusion pipeline
    pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(args.sd_model)
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)

    provider = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    onnx_pipeline = OnnxStableDiffusionControlNetImg2ImgPipeline(
        vae_encoder=OnnxRuntimeModel.from_pretrained(
            os.path.join(args.onnx_model_dir, "vae_encoder"), provider=provider
        ),
        vae_decoder=OnnxRuntimeModel.from_pretrained(
            os.path.join(args.onnx_model_dir, "vae_decoder"), provider=provider
        ),
        text_encoder=OnnxRuntimeModel.from_pretrained(
            os.path.join(args.onnx_model_dir, "text_encoder"), provider=provider
        ),
        tokenizer=pipeline.tokenizer,
        unet=OnnxRuntimeModel.from_pretrained(os.path.join(args.onnx_model_dir, "unet"), provider=provider),
        scheduler=pipeline.scheduler,
    )
    onnx_pipeline = onnx_pipeline.to("cuda")

    prompt = "a cute cat fly to the moon"
    negative_prompt = "paintings, sketches, worst quality, low quality, normal quality, lowres, normal quality, monochrome, grayscale, skin spots, acnes, skin blemishes, age spot, glans, nsfw, nipples, necklace, worst quality, low quality, watermark, username, signature, multiple breasts, lowres, bad anatomy, bad hands, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, bad feet, single color, ugly, duplicate, morbid, mutilated, tranny, trans, trannsexual, hermaphrodite, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, ugly, blurry, bad anatomy, bad proportions, extra limbs, disfigured, bad anatomy, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, mutated hands, fused fingers, too many fingers, long neck, bad body perspect"

    for i in range(10):
        start_time = time.time()
        image = onnx_pipeline(
            num_controlnet=2,
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=qr_image,
            control_image=[qr_image, qr_image],
            width=512,
            height=512,
            strength=0.75,
            num_inference_steps=20,
            num_images_per_prompt=1,
            controlnet_conditioning_scale=[0.8, 0.8],
            control_guidance_start=[0.3, 0.3],
            control_guidance_end=[0.9, 0.9],
        ).images[0]
        print(time.time() - start_time)
        image.save("output_qr_code.png")
