<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Würstchen

<img src="https://github.com/dome272/Wuerstchen/assets/61938694/0617c863-165a-43ee-9303-2a17299a0cf9">

[Wuerstchen: An Efficient Architecture for Large-Scale Text-to-Image Diffusion Models](https://huggingface.co/papers/2306.00637) is by Pablo Pernias, Dominic Rampas, Mats L. Richter and Christopher Pal and Marc Aubreville.

The abstract from the paper is:

*We introduce Würstchen, a novel architecture for text-to-image synthesis that combines competitive performance with unprecedented cost-effectiveness for large-scale text-to-image diffusion models. A key contribution of our work is to develop a latent diffusion technique in which we learn a detailed but extremely compact semantic image representation used to guide the diffusion process. This highly compressed representation of an image provides much more detailed guidance compared to latent representations of language and this significantly reduces the computational requirements to achieve state-of-the-art results. Our approach also improves the quality of text-conditioned image generation based on our user preference study. The training requirements of our approach consists of 24,602 A100-GPU hours - compared to Stable Diffusion 2.1's 200,000 GPU hours. Our approach also requires less training data to achieve these results. Furthermore, our compact latent representations allows us to perform inference over twice as fast, slashing the usual costs and carbon footprint of a state-of-the-art (SOTA) diffusion model significantly, without compromising the end performance. In a broader comparison against SOTA models our approach is substantially more efficient and compares favorably in terms of image quality. We believe that this work motivates more emphasis on the prioritization of both performance and computational accessibility.*

## Würstchen Overview
Würstchen is a diffusion model, whose text-conditional model works in a highly compressed latent space of images. Why is this important? Compressing data can reduce computational costs for both training and inference by magnitudes. Training on 1024x1024 images is way more expensive than training on 32x32. Usually, other works make use of a relatively small compression, in the range of 4x - 8x spatial compression. Würstchen takes this to an extreme. Through its novel design, we achieve a 42x spatial compression. This was unseen before because common methods fail to faithfully reconstruct detailed images after 16x spatial compression. Würstchen employs a two-stage compression, what we call Stage A and Stage B. Stage A is a VQGAN, and Stage B is a Diffusion Autoencoder (more details can be found in the [paper](https://huggingface.co/papers/2306.00637)). A third model, Stage C, is learned in that highly compressed latent space. This training requires fractions of the compute used for current top-performing models, while also allowing cheaper and faster inference.

## Würstchen v2 comes to Diffusers

After the initial paper release, we have improved numerous things in the architecture, training and sampling, making Würstchen competitive to current state-of-the-art models in many ways. We are excited to release this new version together with Diffusers. Here is a list of the improvements.

- Higher resolution (1024x1024 up to 2048x2048)
- Faster inference
- Multi Aspect Resolution Sampling
- Better quality


We are releasing 3 checkpoints for the text-conditional image generation model (Stage C). Those are:

- v2-base
- v2-aesthetic
- **(default)** v2-interpolated (50% interpolation between v2-base and v2-aesthetic)

We recommend using v2-interpolated, as it has a nice touch of both photorealism and aesthetics. Use v2-base for finetunings as it does not have a style bias and use v2-aesthetic for very artistic generations.
A comparison can be seen here:

<img src="https://github.com/dome272/Wuerstchen/assets/61938694/2914830f-cbd3-461c-be64-d50734f4b49d" width=500>

## Text-to-Image Generation

For the sake of usability, Würstchen can be used with a single pipeline. This pipeline can be used as follows:

```python
import torch
from diffusers import AutoPipelineForText2Image
from diffusers.pipelines.wuerstchen import DEFAULT_STAGE_C_TIMESTEPS

pipe = AutoPipelineForText2Image.from_pretrained("warp-ai/wuerstchen", torch_dtype=torch.float16).to("cuda")

caption = "Anthropomorphic cat dressed as a fire fighter"
images = pipe(
    caption,
    width=1024,
    height=1536,
    prior_timesteps=DEFAULT_STAGE_C_TIMESTEPS,
    prior_guidance_scale=4.0,
    num_images_per_prompt=2,
).images
```

For explanation purposes, we can also initialize the two main pipelines of Würstchen individually. Würstchen consists of 3 stages: Stage C, Stage B, Stage A. They all have different jobs and work only together. When generating text-conditional images, Stage C will first generate the latents in a very compressed latent space. This is what happens in the `prior_pipeline`. Afterwards, the generated latents will be passed to Stage B, which decompresses the latents into a bigger latent space of a VQGAN. These latents can then be decoded by Stage A, which is a VQGAN, into the pixel-space. Stage B & Stage A are both encapsulated in the `decoder_pipeline`. For more details, take a look at the [paper](https://huggingface.co/papers/2306.00637).

```python
import torch
from diffusers import WuerstchenDecoderPipeline, WuerstchenPriorPipeline
from diffusers.pipelines.wuerstchen import DEFAULT_STAGE_C_TIMESTEPS

device = "cuda"
dtype = torch.float16
num_images_per_prompt = 2

prior_pipeline = WuerstchenPriorPipeline.from_pretrained(
    "warp-ai/wuerstchen-prior", torch_dtype=dtype
).to(device)
decoder_pipeline = WuerstchenDecoderPipeline.from_pretrained(
    "warp-ai/wuerstchen", torch_dtype=dtype
).to(device)

caption = "Anthropomorphic cat dressed as a fire fighter"
negative_prompt = ""

prior_output = prior_pipeline(
    prompt=caption,
    height=1024,
    width=1536,
    timesteps=DEFAULT_STAGE_C_TIMESTEPS,
    negative_prompt=negative_prompt,
    guidance_scale=4.0,
    num_images_per_prompt=num_images_per_prompt,
)
decoder_output = decoder_pipeline(
    image_embeddings=prior_output.image_embeddings,
    prompt=caption,
    negative_prompt=negative_prompt,
    guidance_scale=0.0,
    output_type="pil",
).images[0]
decoder_output
```

## Speed-Up Inference
You can make use of `torch.compile` function and gain a speed-up of about 2-3x:

```python
prior_pipeline.prior = torch.compile(prior_pipeline.prior, mode="reduce-overhead", fullgraph=True)
decoder_pipeline.decoder = torch.compile(decoder_pipeline.decoder, mode="reduce-overhead", fullgraph=True)
```

## Limitations

- Due to the high compression employed by Würstchen, generations can lack a good amount
of detail. To our human eye, this is especially noticeable in faces, hands etc.
- **Images can only be generated in 128-pixel steps**, e.g. the next higher resolution
after 1024x1024 is 1152x1152
- The model lacks the ability to render correct text in images
- The model often does not achieve photorealism
- Difficult compositional prompts are hard for the model

The original codebase, as well as experimental ideas, can be found at [dome272/Wuerstchen](https://github.com/dome272/Wuerstchen).


## WuerstchenCombinedPipeline

[[autodoc]] WuerstchenCombinedPipeline
	- all
	- __call__

## WuerstchenPriorPipeline

[[autodoc]] WuerstchenPriorPipeline
	- all
	- __call__

## WuerstchenPriorPipelineOutput

[[autodoc]] pipelines.wuerstchen.pipeline_wuerstchen_prior.WuerstchenPriorPipelineOutput

## WuerstchenDecoderPipeline

[[autodoc]] WuerstchenDecoderPipeline
	- all
	- __call__

## Citation

```bibtex
      @misc{pernias2023wuerstchen,
            title={Wuerstchen: An Efficient Architecture for Large-Scale Text-to-Image Diffusion Models},
            author={Pablo Pernias and Dominic Rampas and Mats L. Richter and Christopher J. Pal and Marc Aubreville},
            year={2023},
            eprint={2306.00637},
            archivePrefix={arXiv},
            primaryClass={cs.CV}
      }
```
