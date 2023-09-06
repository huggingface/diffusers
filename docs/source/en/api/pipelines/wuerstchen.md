# Würstchen

<img src="https://github.com/dome272/Wuerstchen/assets/61938694/0617c863-165a-43ee-9303-2a17299a0cf9">

[Würstchen: Efficient Pretraining of Text-to-Image Models](https://huggingface.co/papers/2306.00637) is by Pablo Pernias, Dominic Rampas, and Marc Aubreville.

The abstract from the paper is:

*We introduce Würstchen, a novel technique for text-to-image synthesis that unites competitive performance with unprecedented cost-effectiveness and ease of training on constrained hardware. Building on recent advancements in machine learning, our approach, which utilizes latent diffusion strategies at strong latent image compression rates, significantly reduces the computational burden, typically associated with state-of-the-art models, while preserving, if not enhancing, the quality of generated images. Wuerstchen achieves notable speed improvements at inference time, thereby rendering real-time applications more viable. One of the key advantages of our method lies in its modest training requirements of only 9,200 GPU hours, slashing the usual costs significantly without compromising the end performance. In a comparison against the state-of-the-art, we found the approach to yield strong competitiveness. This paper opens the door to a new line of research that prioritizes both performance and computational accessibility, hence democratizing the use of sophisticated AI technologies. Through Wuerstchen, we demonstrate a compelling stride forward in the realm of text-to-image synthesis, offering an innovative path to explore in future research.*

## Würstchen v2 comes to Diffusers

After the initial paper release, we have improved numerous things in the architecture, training and sampling, making Würstchen competetive to current state-of-the-art models in many ways. We are excited to release this new version together with Diffusers. Here is a list of the improvements.

- Higher resolution (1024x1024 up to 2048x2048)
- Faster inference
- Multi Aspect Resolution Sampling
- Better quality

We are releasing 3 checkpoints for the text-conditional image generation model (Stage C). Those are: 
- v2-base
- v2-aesthetic
- v2-interpolated (50% interpolation between v2-base and v2-aesthetic)

We recommend to use v2-interpolated, as it has a nice touch of both photorealism and aesthetic. Use v2-base for finetunings as it does not have a style bias and use v2-aesthetic for very artistic generations.
A comparison can be seen here: 

<img src="https://github.com/dome272/Wuerstchen/assets/61938694/2914830f-cbd3-461c-be64-d50734f4b49d" width=500>

## Text-to-Image Generation

For the sake of usability Würstchen can be used with a single pipeline. This pipeline is called `WuerstchenCombinedPipeline` and can be used as follows:

```python
import torch
from diffusers import AutoPipelineForText2Image

device = "cuda"
dtype = torch.float16
num_images_per_prompt = 2

pipeline =  AutoPipelineForText2Image.from_pretrained(
    "warp-diffusion/wuerstchen", torch_dtype=dtype
).to(device)

caption = "Anthropomorphic cat dressed as a fire fighter"
negative_prompt = ""

output = pipeline(
    prompt=caption,
    height=1024,
    width=1024,
    negative_prompt=negative_prompt,
    prior_guidance_scale=4.0,
    decoder_guidance_scale=0.0,
    num_images_per_prompt=num_images_per_prompt,
    output_type="pil",
).images
```

For explanation purposes, we can also initialize the two main pipelines of Würstchen individually. Würstchen consists of 3 stages: Stage C, Stage B, Stage A. They all have different jobs and work only together. When generating text-conditional images, Stage C will first generate the latents in a very compressed latent space. This is what happens in the `prior_pipeline`. Afterwards, the generated latents will be passed to Stage B, which decompresses the latents into a bigger latent space of a VQGAN. These latents can then be decoded by Stage A, which is a VQGAN, into the pixel-space. Stage B & Stage A are both encapsulated in the `decoder_pipeline`. For more details, take a look the [paper](https://huggingface.co/papers/2306.00637).

```python
import torch
from diffusers import WuerstchenDecoderPipeline, WuerstchenPriorPipeline

device = "cuda"
dtype = torch.float16
num_images_per_prompt = 2

prior_pipeline = WuerstchenPriorPipeline.from_pretrained(
    "warp-diffusion/wuerstchen-prior", torch_dtype=dtype
).to(device)
decoder_pipeline = WuerstchenDecoderPipeline.from_pretrained(
    "warp-diffusion/wuerstchen", torch_dtype=dtype
).to(device)

caption = "A captivating artwork of a mysterious stone golem"
negative_prompt = ""

prior_output = prior_pipeline(
    prompt=caption,
    height=1024,
    width=1024,
    negative_prompt=negative_prompt,
	guidance_scale=4.0,
    num_images_per_prompt=num_images_per_prompt,
)
decoder_output = decoder_pipeline(
    image_embeddings=prior_output.image_embeddings,
    prompt=caption,
    negative_prompt=negative_prompt,
    num_images_per_prompt=num_images_per_prompt,
    guidance_scale=0.0,
    output_type="pil",
).images
```

## Speed-Up Inference
You can make use of ``torch.compile`` function and gain a speed-up of about 2-3x:

```python
pipeline.prior = torch.compile(pipeline.prior, mode="reduce-overhead", fullgraph=True)
pipeline.decoder = torch.compile(pipeline.decoder, mode="reduce-overhead", fullgraph=True)
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

## WuerschenPipeline

[[autodoc]] WuerstchenCombinedPipeline
	- all
	- __call__

## WuerstchenPriorPipeline

[[autodoc]] WuerstchenDecoderPipeline

	- all
	- __call__

## WuerstchenPriorPipelineOutput

[[autodoc]] pipelines.wuerstchen.pipeline_wuerstchen_prior.WuerstchenPriorPipelineOutput

## WuerstchenDecoderPipeline

[[autodoc]] WuerstchenDecoderPipeline
	- all
	- __call__
