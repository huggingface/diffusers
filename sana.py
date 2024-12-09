import torch

from diffusers import (
    SanaTransformer2DModel, PixArtSigmaPipeline, SanaPipeline, 
    FlowDPMSolverMultistepScheduler, DPMSolverMultistepScheduler
)

from torchvision.utils import save_image



def run_pixart_sigam():
    pipe = PixArtSigmaPipeline.from_pretrained(
        "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
        torch_dtype=torch.float16,
        use_safetensors=True
    ).to(device)


    image = pipe("a dog").images[0]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16
    prompt = 'a cyberpunk cat with a neon sign that says "Sana"'
    generator = torch.Generator(device=device).manual_seed(42)

    latents = torch.randn(
        1,
        32,
        32,
        32,
        generator=generator,
        device=device,
    ).to(dtype)
    # diffusers Sana Model
    pipe = SanaPipeline.from_pretrained(
       "/home/junsongc/junsongc/code/diffusion/Sana/output/Sana_1600M_1024px_diffusers_flowshift3_full", 
       torch_dtype=dtype,
        use_safetensors=True,
    ).to(device)

    scheduler = FlowDPMSolverMultistepScheduler(flow_shift=3.0)
    # scheduler = DPMSolverMultistepScheduler(use_flow_sigmas=True, prediction_type='flow_prediction', flow_shift=3.0)
    pipe.scheduler = scheduler

    pipe.text_encoder.to(torch.bfloat16)
    pipe.vae.to(torch.float32)
    image = pipe(
        prompt=prompt,
        height=1024,
        width=1024,
        guidance_scale=5.0,
        num_inference_steps=20,
        generator=generator,
        latents=latents,
    )[0]
    image[0].save('sana_diffusers_flowdpmsolver4.png')


    # run pixart
    # run_pixart_sigam()
