import torch
from diffusers import WuerstchenPriorPipeline

prior_pipeline = WuerstchenPriorPipeline.from_pretrained("C:\\Users\\d6582\\Documents\\ml\\diffusers\\scripts\\kashif\\WuerstchenPriorPipeline", torch_dtype=torch.float16)
prior_pipeline = prior_pipeline.to("cuda")

generator_pipeline = WuerstchenPriorPipeline.from_pretrained("C:\\Users\\d6582\\Documents\\ml\\diffusers\\scripts\\kashif\\WuerstchenPriorPipeline", torch_dtype=torch.float16)
generator_pipeline = generator_pipeline.to("cuda")

generator_output = generator_pipeline("An image of a squirrel in Picasso style")