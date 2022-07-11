import torch

from ...models import UNetLDMModel, VQModel
from ...pipeline_utils import DiffusionPipeline
from ...schedulers import DiscreteScheduler


class LatentDiffusionUncondPipeline(DiffusionPipeline):
    unet: UNetLDMModel
    vqvae: VQModel
    noise_scheduler: DiscreteScheduler

    def __init__(self, vqvae, unet, noise_scheduler):
        super().__init__()
        noise_scheduler = noise_scheduler.set_format("pt")
        self.register_modules(vqvae=vqvae, unet=unet, noise_scheduler=noise_scheduler)

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        eta: float = 0.0,
        num_inference_steps: int = None,
        seed: int = None,
        device: str = None,
    ):
        if num_inference_steps is None:
            num_inference_steps = self.noise_scheduler.num_timesteps
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        random_generator = torch.manual_seed(seed)
        self.unet.to(device)
        self.vqvae.to(device)

        latents = torch.randn(
            (batch_size, self.unet.in_channels, self.unet.image_size, self.unet.image_size),
            generator=random_generator,
        )
        latents = latents.to(device)

        self.noise_scheduler.set_num_inference_steps(num_inference_steps)

        for t in reversed(range(num_inference_steps)):
            # adjust the reduced timestep to the number of training timesteps
            t = t * (self.noise_scheduler.num_timesteps // num_inference_steps)
            noise_prediction = self.unet(latents, t)
            noise = torch.randn(latents.shape, generator=random_generator).to(device)
            latents = self.noise_scheduler.step(noise_prediction, latents, t, eta=eta, noise=noise)

        # decode the image latents with the VAE
        image = self.vqvae.decode(latents)
        image = (image / 2 + 0.5).cpu().permute(0, 2, 3, 1).numpy()

        return image
