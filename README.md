<p align="center">
    <br>
    <img src="./docs/source/en/imgs/diffusers_library.jpg" width="400"/>
    <br>
<p>
<p align="center">
    <a href="https://github.com/huggingface/diffusers/blob/main/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/huggingface/datasets.svg?color=blue">
    </a>
    <a href="https://github.com/huggingface/diffusers/releases">
        <img alt="GitHub release" src="https://img.shields.io/github/release/huggingface/diffusers.svg">
    </a>
    <a href="CODE_OF_CONDUCT.md">
        <img alt="Contributor Covenant" src="https://img.shields.io/badge/Contributor%20Covenant-2.0-4baaaa.svg">
    </a>
</p>

🤗 Diffusers is the go-to library for state-of-the-art pretrained diffusion models for generating images, audio, and even 3D structures of molecules. Whether you're looking for a simple inference solution or training your own diffusion models, 🤗 Diffusers is a modular toolbox that supports both. Our library is designed with a focus on [usability over performance](https://huggingface.co/docs/diffusers/conceptual/philosophy#usability-over-performance), [simple over easy](https://huggingface.co/docs/diffusers/conceptual/philosophy#simple-over-easy), and [customizability over abstractions](https://huggingface.co/docs/diffusers/conceptual/philosophy#tweakable-contributorfriendly-over-abstraction).

🤗 Diffusers offers three core components:

- State-of-the-art [diffusion pipelines](https://huggingface.co/docs/diffusers/api/pipelines/overview) that can be run in inference with just a few lines of code.
- Interchangeable noise [schedulers](https://huggingface.co/docs/diffusers/api/schedulers/overview) for different diffusion speeds and output quality.
- Pretrained [models](https://huggingface.co/docs/diffusers/api/models) that can be used as building blocks, and combined with schedulers, for creating your own end-to-end diffusion systems.

## Installation

We recommend installing 🤗 Diffusers in a virtual environment from PyPi or Conda. For more details about installing [PyTorch](https://pytorch.org/get-started/locally/) and [Flax](https://flax.readthedocs.io/en/latest/installation.html), please refer to their official documentation.

### PyTorch

With `pip` (official package):
    
```bash
pip install --upgrade diffusers[torch]
```

With `conda` (maintained by the community):

```sh
conda install -c conda-forge diffusers
```

### Flax

With `pip` (official package):

```bash
pip install --upgrade diffusers[flax]
```

### Apple Silicon (M1/M2) support

Please refer to the [How to use Stable Diffusion in Apple Silicon](https://huggingface.co/docs/diffusers/optimization/mps) guide.

## Quickstart

Generating outputs is super easy with 🤗 Diffusers. To generate an image from text, use the `from_pretrained` method to load any pretrained diffusion model (browse the [Hub](https://huggingface.co/models?library=diffusers&sort=downloads) for 4000+ checkpoints):

```python
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipeline.to("cuda")
pipeline("An image of a squirrel in Picasso style").images[0]
```

You can also dig into the models and schedulers toolbox to build your own diffusion system:

```python
from diffusers import DDPMScheduler, UNet2DModel
from PIL import Image
import torch
import numpy as np

scheduler = DDPMScheduler.from_pretrained("google/ddpm-cat-256")
model = UNet2DModel.from_pretrained("google/ddpm-cat-256").to("cuda")
scheduler.set_timesteps(50)

sample_size = model.config.sample_size
noise = torch.randn((1, 3, sample_size, sample_size)).to("cuda")
input = noise

for t in scheduler.timesteps:
    with torch.no_grad():
        noisy_residual = model(input, t).sample
        prev_noisy_sample = scheduler.step(noisy_residual, t, input).prev_sample
        input = prev_noisy_sample

image = (input / 2 + 0.5).clamp(0, 1)
image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
image = Image.fromarray((image * 255).round().astype("uint8"))
image
```

Check out the [Quickstart](https://huggingface.co/docs/diffusers/quicktour) to launch your diffusion journey today!

## How to navigate the documentation

| **Documentation**                                                   | **What can I learn?**                                                                                                                                                                           |
|---------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Tutorial                                                            | A basic crash course for learning how to use the library's most important features like using models and schedulers to build your own diffusion system, and training your own diffusion model.  |
| Loading                                                             | Guides for how to load and configure all the components (pipelines, models, and schedulers) of the library, as well as how to use different schedulers.                                         |
| Pipelines for inference                                             | Guides for how to use pipelines for different inference tasks, batched generation, controlling generated outputs and randomness, and how to contribute a pipeline to the library.               |
| Optimization                                                        | Guides for how to optimize your diffusion model to run faster and consume less memory.                                                                                                          |
| [Training](https://huggingface.co/docs/diffusers/training/overview) | Guides for how to train a diffusion model for different tasks with different training techniques.                                                                                               |

## Supported pipelines

| Pipeline | Paper | Tasks |
|---|---|:---:|
| [alt_diffusion](./api/pipelines/alt_diffusion) | [**AltDiffusion**](https://arxiv.org/abs/2211.06679) | Image-to-Image Text-Guided Generation |
| [audio_diffusion](./api/pipelines/audio_diffusion) | [**Audio Diffusion**](https://github.com/teticio/audio-diffusion.git) | Unconditional Audio Generation |
| [controlnet](./api/pipelines/stable_diffusion/controlnet) | [**ControlNet with Stable Diffusion**](https://arxiv.org/abs/2302.05543) | Image-to-Image Text-Guided Generation |
| [cycle_diffusion](./api/pipelines/cycle_diffusion) | [**Cycle Diffusion**](https://arxiv.org/abs/2210.05559) | Image-to-Image Text-Guided Generation |
| [dance_diffusion](./api/pipelines/dance_diffusion) | [**Dance Diffusion**](https://github.com/williamberman/diffusers.git) | Unconditional Audio Generation |
| [ddpm](./api/pipelines/ddpm) | [**Denoising Diffusion Probabilistic Models**](https://arxiv.org/abs/2006.11239) | Unconditional Image Generation |
| [ddim](./api/pipelines/ddim) | [**Denoising Diffusion Implicit Models**](https://arxiv.org/abs/2010.02502) | Unconditional Image Generation |
| [latent_diffusion](./api/pipelines/latent_diffusion) | [**High-Resolution Image Synthesis with Latent Diffusion Models**](https://arxiv.org/abs/2112.10752)| Text-to-Image Generation |
| [latent_diffusion](./api/pipelines/latent_diffusion) | [**High-Resolution Image Synthesis with Latent Diffusion Models**](https://arxiv.org/abs/2112.10752)| Super Resolution Image-to-Image |
| [latent_diffusion_uncond](./api/pipelines/latent_diffusion_uncond) | [**High-Resolution Image Synthesis with Latent Diffusion Models**](https://arxiv.org/abs/2112.10752) | Unconditional Image Generation |
| [paint_by_example](./api/pipelines/paint_by_example) | [**Paint by Example: Exemplar-based Image Editing with Diffusion Models**](https://arxiv.org/abs/2211.13227) | Image-Guided Image Inpainting |
| [pndm](./api/pipelines/pndm) | [**Pseudo Numerical Methods for Diffusion Models on Manifolds**](https://arxiv.org/abs/2202.09778) | Unconditional Image Generation |
| [score_sde_ve](./api/pipelines/score_sde_ve) | [**Score-Based Generative Modeling through Stochastic Differential Equations**](https://openreview.net/forum?id=PxTIG12RRHS) | Unconditional Image Generation |
| [score_sde_vp](./api/pipelines/score_sde_vp) | [**Score-Based Generative Modeling through Stochastic Differential Equations**](https://openreview.net/forum?id=PxTIG12RRHS) | Unconditional Image Generation |
| [semantic_stable_diffusion](./api/pipelines/semantic_stable_diffusion) | [**Semantic Guidance**](https://arxiv.org/abs/2301.12247) | Text-Guided Generation |
| [stable_diffusion_text2img](./api/pipelines/stable_diffusion/text2img) | [**Stable Diffusion**](https://stability.ai/blog/stable-diffusion-public-release) | Text-to-Image Generation |
| [stable_diffusion_img2img](./api/pipelines/stable_diffusion/img2img) | [**Stable Diffusion**](https://stability.ai/blog/stable-diffusion-public-release) | Image-to-Image Text-Guided Generation |
| [stable_diffusion_inpaint](./api/pipelines/stable_diffusion/inpaint) | [**Stable Diffusion**](https://stability.ai/blog/stable-diffusion-public-release) | Text-Guided Image Inpainting |
| [stable_diffusion_panorama](./api/pipelines/stable_diffusion/panorama) | [**MultiDiffusion**](https://multidiffusion.github.io/) | Text-to-Panorama Generation |
| [stable_diffusion_pix2pix](./api/pipelines/stable_diffusion/pix2pix) | [**InstructPix2Pix**](https://github.com/timothybrooks/instruct-pix2pix) | Text-Guided Image Editing|
| [stable_diffusion_pix2pix_zero](./api/pipelines/stable_diffusion/pix2pix_zero) | [**Zero-shot Image-to-Image Translation**](https://pix2pixzero.github.io/) | Text-Guided Image Editing |
| [stable_diffusion_attend_and_excite](./api/pipelines/stable_diffusion/attend_and_excite) | [**Attend and Excite for Stable Diffusion**](https://attendandexcite.github.io/Attend-and-Excite/) | Text-to-Image Generation |
| [stable_diffusion_self_attention_guidance](./api/pipelines/stable_diffusion/self_attention_guidance) | [**Self-Attention Guidance**](https://ku-cvlab.github.io/Self-Attention-Guidance) | Text-to-Image Generation |
| [stable_diffusion_image_variation](./stable_diffusion/image_variation) | [**Stable Diffusion Image Variations**](https://github.com/LambdaLabsML/lambda-diffusers#stable-diffusion-image-variations) | Image-to-Image Generation |
| [stable_diffusion_latent_upscale](./stable_diffusion/latent_upscale) | [**Stable Diffusion Latent Upscaler**](https://twitter.com/StabilityAI/status/1590531958815064065) | Text-Guided Super Resolution Image-to-Image |
| [stable_diffusion_2](./api/pipelines/stable_diffusion_2) | [**Stable Diffusion 2**](https://stability.ai/blog/stable-diffusion-v2-release) | Text-to-Image Generation |
| [stable_diffusion_2](./api/pipelines/stable_diffusion_2) | [**Stable Diffusion 2**](https://stability.ai/blog/stable-diffusion-v2-release) | Text-Guided Image Inpainting |
| [stable_diffusion_2](./api/pipelines/stable_diffusion_2) | [**Depth-Conditional Stable Diffusion**](https://github.com/Stability-AI/stablediffusion#depth-conditional-stable-diffusion) | Depth-to-Image Generation |
| [stable_diffusion_2](./api/pipelines/stable_diffusion_2) | [**Stable Diffusion 2**](https://stability.ai/blog/stable-diffusion-v2-release) | Text-Guided Super Resolution Image-to-Image |
| [stable_diffusion_safe](./api/pipelines/stable_diffusion_safe) | [**Safe Stable Diffusion**](https://arxiv.org/abs/2211.05105) | Text-Guided Generation |
| [stable_unclip](./stable_unclip) | **Stable unCLIP** | Text-to-Image Generation |
| [stable_unclip](./stable_unclip) | **Stable unCLIP** | Image-to-Image Text-Guided Generation |
| [stochastic_karras_ve](./api/pipelines/stochastic_karras_ve) | [**Elucidating the Design Space of Diffusion-Based Generative Models**](https://arxiv.org/abs/2206.00364) | Unconditional Image Generation |
| [unclip](./api/pipelines/unclip) | [Hierarchical Text-Conditional Image Generation with CLIP Latents](https://arxiv.org/abs/2204.06125) | Text-to-Image Generation |
| [versatile_diffusion](./api/pipelines/versatile_diffusion) | [Versatile Diffusion: Text, Images and Variations All in One Diffusion Model](https://arxiv.org/abs/2211.08332) | Text-to-Image Generation |
| [versatile_diffusion](./api/pipelines/versatile_diffusion) | [Versatile Diffusion: Text, Images and Variations All in One Diffusion Model](https://arxiv.org/abs/2211.08332) | Image Variations Generation |
| [versatile_diffusion](./api/pipelines/versatile_diffusion) | [Versatile Diffusion: Text, Images and Variations All in One Diffusion Model](https://arxiv.org/abs/2211.08332) | Dual Image and Text Guided Generation |
| [vq_diffusion](./api/pipelines/vq_diffusion) | [Vector Quantized Diffusion Model for Text-to-Image Synthesis](https://arxiv.org/abs/2111.14822) | Text-to-Image Generation |

## Contribution

We ❤️  contributions from the open-source community! 
If you want to contribute to this library, please check out our [Contribution guide](https://github.com/huggingface/diffusers/blob/main/CONTRIBUTING.md).
You can look out for [issues](https://github.com/huggingface/diffusers/issues) you'd like to tackle to contribute to the library.
- See [Good first issues](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22) for general opportunities to contribute
- See [New model/pipeline](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22New+pipeline%2Fmodel%22) to contribute exciting new diffusion models / diffusion pipelines
- See [New scheduler](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22New+scheduler%22)

Also, say 👋 in our public Discord channel <a href="https://discord.gg/G7tWnz98XR"><img alt="Join us on Discord" src="https://img.shields.io/discord/823813159592001537?color=5865F2&logo=discord&logoColor=white"></a>. We discuss the hottest trends about diffusion models, help each other with contributions, personal projects or
just hang out ☕.

## Credits

This library concretizes previous work by many different authors and would not have been possible without their great research and implementations. We'd like to thank, in particular, the following implementations which have helped us in our development and without which the API could not have been as polished today:

- @CompVis' latent diffusion models library, available [here](https://github.com/CompVis/latent-diffusion)
- @hojonathanho original DDPM implementation, available [here](https://github.com/hojonathanho/diffusion) as well as the extremely useful translation into PyTorch by @pesser, available [here](https://github.com/pesser/pytorch_diffusion)
- @ermongroup's DDIM implementation, available [here](https://github.com/ermongroup/ddim)
- @yang-song's Score-VE and Score-VP implementations, available [here](https://github.com/yang-song/score_sde_pytorch)

We also want to thank @heejkoo for the very helpful overview of papers, code and resources on diffusion models, available [here](https://github.com/heejkoo/Awesome-Diffusion-Models) as well as @crowsonkb and @rromb for useful discussions and insights.

## Citation

```bibtex
@misc{von-platen-etal-2022-diffusers,
  author = {Patrick von Platen and Suraj Patil and Anton Lozhkov and Pedro Cuenca and Nathan Lambert and Kashif Rasul and Mishig Davaadorj and Thomas Wolf},
  title = {Diffusers: State-of-the-art diffusion models},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/huggingface/diffusers}}
}
```
