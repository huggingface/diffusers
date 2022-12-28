from typing import List, Optional, Tuple, Union

import torch

from ...models import AutoencoderKL, DiT
from ...pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from ...schedulers import DDIMScheduler


class DiTPipeline(DiffusionPipeline):
    def __init__(self, dit: DiT, vae: AutoencoderKL, scheduler: DDIMScheduler):
        super().__init__()
        self.register_modules(dit=dit, vae=vae, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
        self,
        class_labels: List[int],
        guidance_scale: float = 4.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 250,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        **kwargs,
    ) -> Union[ImagePipelineOutput, Tuple]:
        batch_size = len(class_labels)
        latent_size = self.dit.config.input_size

        latents = torch.randn(batch_size, 4, latent_size, latent_size, device=self.device)
        latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1 else latents

        class_labels = torch.tensor(class_labels, device=self.device)
        class_null = torch.tensor([1000] * batch_size, device=self.device)
        class_labels_input = torch.cat([class_labels, class_null], 0) if guidance_scale > 1 else class_labels

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.progress_bar(self.scheduler.timesteps):
            if guidance_scale > 1:
                half = latent_model_input[: len(latent_model_input) // 2]
                latent_model_input = torch.cat([half, half], dim=0)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict noise model_output
            noise_pred = self.dit(latent_model_input, t, class_labels_input)

            # perform guidance
            if guidance_scale > 1:
                eps, rest = noise_pred[:, :3], noise_pred[:, 3:]
                cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)

                half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
                eps = torch.cat([half_eps, half_eps], dim=0)

                noise_pred = torch.cat([eps, rest], dim=1)

            # learned sigma
            _, C = latent_model_input.shape[:2]
            model_output, _ = torch.split(noise_pred, C, dim=1)

            # compute previous image: x_t -> x_t-1
            latent_model_input = self.scheduler.step(
                model_output, t, latent_model_input, generator=generator
            ).prev_sample

        if guidance_scale > 1:
            latents, _ = latent_model_input.chunk(2, dim=0)

        latents = 1 / 0.18215 * latents
        samples = self.vae.decode(latents).sample

        samples = (samples / 2 + 0.5).clamp(0, 1)

        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        samples = samples.cpu().permute(0, 2, 3, 1).float().numpy()

        if output_type == "pil":
            samples = self.numpy_to_pil(samples)

        if not return_dict:
            return (samples,)

        return ImagePipelineOutput(images=samples)
