import torch

from tqdm.auto import tqdm

from ...pipeline_utils import DiffusionPipeline


class LatentDiffusionUncondPipeline(DiffusionPipeline):
    def __init__(self, vqvae, unet, scheduler):
        super().__init__()
        scheduler = scheduler.set_format("pt")
        self.register_modules(vqvae=vqvae, unet=unet, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
        self, batch_size=1, generator=None, torch_device=None, eta=0.0, num_inference_steps=50, output_type="numpy"
    ):
        # eta corresponds to η in paper and should be between [0, 1]

        if torch_device is None:
            torch_device = "cuda" if torch.cuda.is_available() else "cpu"

        self.unet.to(torch_device)
        self.vqvae.to(torch_device)

        latents = torch.randn(
            (batch_size, self.unet.in_channels, self.unet.image_size, self.unet.image_size),
            generator=generator,
        )
        latents = latents.to(torch_device)

        self.scheduler.set_timesteps(num_inference_steps)

        for t in tqdm(self.scheduler.timesteps):
            # predict the noise residual
            noise_prediction = self.unet(latents, t)["sample"]
            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_prediction, t, latents, eta)["prev_sample"]

        # decode the image latents with the VAE
        image = self.vqvae.decode(latents)

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        return {"sample": image}
