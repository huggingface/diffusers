from diffusers import AbsorbingDiffusionPipeline, Transformer, VQModel


# 1. create VQ-VAE
vae = VQModel.from_pretrained("/Users/nielsrogge/Documents/AbsorbingDiffusion/churches/test/vae")

# 2. create Transformer
transformer = Transformer.from_pretrained("/Users/nielsrogge/Documents/AbsorbingDiffusion/churches/test/transformer")

# 3. create pipeline
pipe = AbsorbingDiffusionPipeline(vae=vae, transformer=transformer)

# 4. save the pipeline
pipe.save_pretrained("/Users/nielsrogge/Documents/AbsorbingDiffusion/churches/test")

# generate images
# pipe(batch_size=1, height=512, width=512, num_inference_steps=256)
