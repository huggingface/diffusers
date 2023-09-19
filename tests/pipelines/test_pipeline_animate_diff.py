from diffusers import AutoencoderKL
from diffusers import DDIMScheduler, EulerDiscreteScheduler, PNDMScheduler
from diffusers.utils.import_utils import is_xformers_available
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers import AnimateDiffPipeline
from diffusers import AnimateDiffUNet3DConditionModel


def main():
    tokenizer = CLIPTokenizer.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="text_encoder").cuda()
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae").cuda()
    unet = AnimateDiffUNet3DConditionModel.from_pretrained_2d("CompVis/stable-diffusion-v1-4", subfolder="unet").cuda()
    pipeline = AnimateDiffPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet, scheduler=DDIMScheduler
    ).to("cuda")

    # NOTE: test loading of pipeline


if __name__ == "__main__":
    main()
