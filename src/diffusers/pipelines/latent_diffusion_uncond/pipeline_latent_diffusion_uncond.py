import torch

import tqdm

from ...pipeline_utils import DiffusionPipeline


class LatentDiffusionUncondPipeline(DiffusionPipeline):
    def __init__(self, vqvae, unet, scheduler):
        super().__init__()
        scheduler = scheduler.set_format("pt")
        self.register_modules(vqvae=vqvae, unet=unet, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
        self,
        batch_size=1,
        generator=None,
        torch_device=None,
        eta=0.0,
        num_inference_steps=50,
    ):
        # eta corresponds to η in paper and should be between [0, 1]

        if torch_device is None:
            torch_device = "cuda" if torch.cuda.is_available() else "cpu"

        self.unet.to(torch_device)
        self.vqvae.to(torch_device)

        image = torch.randn(
            (batch_size, self.unet.in_channels, self.unet.image_size, self.unet.image_size),
            generator=generator,
        ).to(torch_device)

        self.scheduler.set_timesteps(num_inference_steps)

        for t in tqdm.tqdm(self.scheduler.timesteps):
            with torch.no_grad():
                model_output = self.unet(image, t)

            if isinstance(model_output, dict):
                model_output = model_output["sample"]

            # 2. predict previous mean of image x_t-1 and add variance depending on eta
            # do x_t -> x_t-1
            image = self.scheduler.step(model_output, t, image, eta)["prev_sample"]

        # decode image with vae
        with torch.no_grad():
            image = self.vqvae.decode(image)
        return {"sample": image}
