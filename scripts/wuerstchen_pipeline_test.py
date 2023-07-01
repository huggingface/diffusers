import torch
from diffusers import WuerstchenPriorPipeline, WuerstchenGeneratorPipeline

prior_pipeline = WuerstchenPriorPipeline.from_pretrained("C:\\Users\\d6582\\Documents\\ml\\diffusers\\scripts\\kashif\\WuerstchenPriorPipeline", torch_dtype=torch.float16)
generator_pipeline = WuerstchenGeneratorPipeline.from_pretrained("C:\\Users\\d6582\\Documents\\ml\\diffusers\\scripts\\kashif\\WuerstchenGeneratorPipeline", torch_dtype=torch.float16)
prior_pipeline = prior_pipeline.to("cuda")
generator_pipeline = generator_pipeline.to("cuda")

prior_output = prior_pipeline("An image of a squirrel in Picasso style")
generator_output = generator_pipeline(prior_output.image_embeds, prior_output.text_embeds)
