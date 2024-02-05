# AsymmetricAutoencoderKL

Improved larger variational autoencoder (VAE) model with KL loss for inpainting task: [Designing a Better Asymmetric VQGAN for StableDiffusion](https://arxiv.org/abs/2306.04632) by Zixin Zhu, Xuelu Feng, Dongdong Chen, Jianmin Bao, Le Wang, Yinpeng Chen, Lu Yuan, Gang Hua.

The abstract from the paper is:

*StableDiffusion is a revolutionary text-to-image generator that is causing a stir in the world of image generation and editing. Unlike traditional methods that learn a diffusion model in pixel space, StableDiffusion learns a diffusion model in the latent space via a VQGAN, ensuring both efficiency and quality. It not only supports image generation tasks, but also enables image editing for real images, such as image inpainting and local editing. However, we have observed that the vanilla VQGAN used in StableDiffusion leads to significant information loss, causing distortion artifacts even in non-edited image regions. To this end, we propose a new asymmetric VQGAN with two simple designs. Firstly, in addition to the input from the encoder, the decoder contains a conditional branch that incorporates information from task-specific priors, such as the unmasked image region in inpainting. Secondly, the decoder is much heavier than the encoder, allowing for more detailed recovery while only slightly increasing the total inference cost. The training cost of our asymmetric VQGAN is cheap, and we only need to retrain a new asymmetric decoder while keeping the vanilla VQGAN encoder and StableDiffusion unchanged. Our asymmetric VQGAN can be widely used in StableDiffusion-based inpainting and local editing methods. Extensive experiments demonstrate that it can significantly improve the inpainting and editing performance, while maintaining the original text-to-image capability. The code is available at https://github.com/buxiangzhiren/Asymmetric_VQGAN*

Evaluation results can be found in section 4.1 of the original paper. 

## Available checkpoints

* [https://huggingface.co/cross-attention/asymmetric-autoencoder-kl-x-1-5](https://huggingface.co/cross-attention/asymmetric-autoencoder-kl-x-1-5)
* [https://huggingface.co/cross-attention/asymmetric-autoencoder-kl-x-2](https://huggingface.co/cross-attention/asymmetric-autoencoder-kl-x-2)

## Example Usage

```python
from io import BytesIO
from PIL import Image
import requests
from diffusers import AsymmetricAutoencoderKL, StableDiffusionInpaintPipeline


def download_image(url: str) -> Image.Image:
    response = requests.get(url)
    return Image.open(BytesIO(response.content)).convert("RGB")


prompt = "a photo of a person"
img_url = "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/repaint/celeba_hq_256.png"
mask_url = "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/repaint/mask_256.png"

image = download_image(img_url).resize((256, 256))
mask_image = download_image(mask_url).resize((256, 256))

pipe = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting")
pipe.vae = AsymmetricAutoencoderKL.from_pretrained("cross-attention/asymmetric-autoencoder-kl-x-1-5")
pipe.to("cuda")

image = pipe(prompt=prompt, image=image, mask_image=mask_image).images[0]
image.save("image.jpeg")
```

## AsymmetricAutoencoderKL

[[autodoc]] models.autoencoder_asym_kl.AsymmetricAutoencoderKL

## AutoencoderKLOutput

[[autodoc]] models.autoencoder_kl.AutoencoderKLOutput

## DecoderOutput

[[autodoc]] models.vae.DecoderOutput
