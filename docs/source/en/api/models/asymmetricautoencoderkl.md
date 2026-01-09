<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# AsymmetricAutoencoderKL

Improved larger variational autoencoder (VAE) model with KL loss for inpainting task: [Designing a Better Asymmetric VQGAN for StableDiffusion](https://huggingface.co/papers/2306.04632) by Zixin Zhu, Xuelu Feng, Dongdong Chen, Jianmin Bao, Le Wang, Yinpeng Chen, Lu Yuan, Gang Hua.

The abstract from the paper is:

*StableDiffusion is a revolutionary text-to-image generator that is causing a stir in the world of image generation and editing. Unlike traditional methods that learn a diffusion model in pixel space, StableDiffusion learns a diffusion model in the latent space via a VQGAN, ensuring both efficiency and quality. It not only supports image generation tasks, but also enables image editing for real images, such as image inpainting and local editing. However, we have observed that the vanilla VQGAN used in StableDiffusion leads to significant information loss, causing distortion artifacts even in non-edited image regions. To this end, we propose a new asymmetric VQGAN with two simple designs. Firstly, in addition to the input from the encoder, the decoder contains a conditional branch that incorporates information from task-specific priors, such as the unmasked image region in inpainting. Secondly, the decoder is much heavier than the encoder, allowing for more detailed recovery while only slightly increasing the total inference cost. The training cost of our asymmetric VQGAN is cheap, and we only need to retrain a new asymmetric decoder while keeping the vanilla VQGAN encoder and StableDiffusion unchanged. Our asymmetric VQGAN can be widely used in StableDiffusion-based inpainting and local editing methods. Extensive experiments demonstrate that it can significantly improve the inpainting and editing performance, while maintaining the original text-to-image capability. The code is available at https://github.com/buxiangzhiren/Asymmetric_VQGAN*

Evaluation results can be found in section 4.1 of the original paper.

## Available checkpoints

* [https://huggingface.co/cross-attention/asymmetric-autoencoder-kl-x-1-5](https://huggingface.co/cross-attention/asymmetric-autoencoder-kl-x-1-5)
* [https://huggingface.co/cross-attention/asymmetric-autoencoder-kl-x-2](https://huggingface.co/cross-attention/asymmetric-autoencoder-kl-x-2)

## Example Usage

```python
from diffusers import AsymmetricAutoencoderKL, StableDiffusionInpaintPipeline
from diffusers.utils import load_image, make_image_grid


prompt = "a photo of a person with beard"
img_url = "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/repaint/celeba_hq_256.png"
mask_url = "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/repaint/mask_256.png"

original_image = load_image(img_url).resize((512, 512))
mask_image = load_image(mask_url).resize((512, 512))

pipe = StableDiffusionInpaintPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-inpainting")
pipe.vae = AsymmetricAutoencoderKL.from_pretrained("cross-attention/asymmetric-autoencoder-kl-x-1-5")
pipe.to("cuda")

image = pipe(prompt=prompt, image=original_image, mask_image=mask_image).images[0]
make_image_grid([original_image, mask_image, image], rows=1, cols=3)
```

## AsymmetricAutoencoderKL

[[autodoc]] models.autoencoders.autoencoder_asym_kl.AsymmetricAutoencoderKL

## AutoencoderKLOutput

[[autodoc]] models.autoencoders.autoencoder_kl.AutoencoderKLOutput

## DecoderOutput

[[autodoc]] models.autoencoders.vae.DecoderOutput
