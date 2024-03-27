from diffusers import DiffusionPipeline, DDIMScheduler, DDPMScheduler

if __name__ == "__main__":
    pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", debug=True)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    #pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("mps")

    prompt = "a cute teddy bear in a cozy scarf"
    image = pipe(prompt).images[0]
    image.save("bear.png")
    print("finish")
