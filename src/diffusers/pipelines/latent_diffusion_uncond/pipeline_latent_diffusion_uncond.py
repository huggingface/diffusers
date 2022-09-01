import inspect
import warnings

import torch

from ...pipeline_utils import DiffusionPipeline


class LDMPipeline(DiffusionPipeline):
    def __init__(self, vqvae, unet, scheduler):
        super().__init__()
        scheduler = scheduler.set_format("pt")
        self.register_modules(vqvae=vqvae, unet=unet, scheduler=scheduler)

    @torch.no_grad()
    def __call__(self, batch_size=1, generator=None, eta=0.0, num_inference_steps=50, output_type="pil", **kwargs):
        # eta corresponds to η in paper and should be between [0, 1]

        if "torch_device" in kwargs:
            device = kwargs.pop("torch_device")
            warnings.warn(
                "`torch_device` is deprecated as an input argument to `__call__` and will be removed in v0.3.0."
                " Consider using `pipe.to(torch_device)` instead."
            )

            # Set device as before (to be removed in 0.3.0)
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.to(device)

        latents = torch.randn(
            (batch_size, self.unet.in_channels, self.unet.sample_size, self.unet.sample_size),
            generator=generator,
        )
        latents = latents.to(self.device)

        self.scheduler.set_timesteps(num_inference_steps)

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())

        extra_kwargs = {}
        if accepts_eta:
            extra_kwargs["eta"] = eta

        for t in self.progress_bar(self.scheduler.timesteps):
            # predict the noise residual
            noise_prediction = self.unet(latents, t)["sample"]
            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_prediction, t, latents, **extra_kwargs)["prev_sample"]

        # decode the image latents with the VAE
        image = self.vqvae.decode(latents)

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        return {"sample": image}
