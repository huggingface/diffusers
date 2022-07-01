import torch

import tqdm

from ...pipeline_utils import DiffusionPipeline


class LatentDiffusionUncondPipeline(DiffusionPipeline):
    def __init__(self, vqvae, unet, noise_scheduler):
        super().__init__()
        noise_scheduler = noise_scheduler.set_format("pt")
        self.register_modules(vqvae=vqvae, unet=unet, noise_scheduler=noise_scheduler)

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

        num_trained_timesteps = self.noise_scheduler.config.timesteps
        inference_step_times = range(0, num_trained_timesteps, num_trained_timesteps // num_inference_steps)

        image = torch.randn(
            (batch_size, self.unet.in_channels, self.unet.image_size, self.unet.image_size),
            generator=generator,
        ).to(torch_device)

        # See formulas (12) and (16) of DDIM paper https://arxiv.org/pdf/2010.02502.pdf
        # Ideally, read DDIM paper in-detail understanding

        # Notation (<variable name> -> <name in paper>
        # - pred_noise_t -> e_theta(x_t, t)
        # - pred_original_image -> f_theta(x_t, t) or x_0
        # - std_dev_t -> sigma_t
        # - eta -> η
        # - pred_image_direction -> "direction pointingc to x_t"
        # - pred_prev_image -> "x_t-1"
        for t in tqdm.tqdm(reversed(range(num_inference_steps)), total=num_inference_steps):
            # 1. predict noise residual
            timesteps = torch.tensor([inference_step_times[t]] * image.shape[0], device=torch_device)
            pred_noise_t = self.unet(image, timesteps)

            # 2. predict previous mean of image x_t-1
            pred_prev_image = self.noise_scheduler.step(pred_noise_t, image, t, num_inference_steps, eta)

            # 3. optionally sample variance
            variance = 0
            if eta > 0:
                noise = torch.randn(image.shape, generator=generator).to(image.device)
                variance = self.noise_scheduler.get_variance(t, num_inference_steps).sqrt() * eta * noise

            # 4. set current image to prev_image: x_t -> x_t-1
            image = pred_prev_image + variance

        # decode image with vae
        image = self.vqvae.decode(image)
        return image
