from typing import List, Optional, Tuple, Union

import torch

from ...configuration_utils import FrozenDict
from ...models import AutoencoderKL, DiT
from ...pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from ...schedulers import DDIMScheduler
from ...utils import deprecate


class DiTPipeline(DiffusionPipeline):
    def __init__(self, dit: DiT, vae: AutoencoderKL, scheduler: DDIMScheduler):
        super().__init__()
        self.register_modules(dit=dit, vae=vae, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
        self,
        class_labels: List[int],
        cfg_scale: float = 4.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 250,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        **kwargs,
    ) -> Union[ImagePipelineOutput, Tuple]:
        batch_size = len(class_labels)
        latent_size = self.dit.config.input_size

        image = torch.randn(batch_size, 4, latent_size, latent_size, device=self.device)
        y = torch.tensor(class_labels, device=self.device)

        # Setup classifier-free guidance:
        image = torch.cat([image, image], 0)
        y_null = torch.tensor([1000] * batch_size, device=self.device)
        y = torch.cat([y, y_null], 0)

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.progress_bar(self.scheduler.timesteps):
            # 1. predict noise model_output
            model_output = self.forward_with_cfg(image, t, y, cfg_scale)

            # 2. compute previous image: x_t -> x_t-1
            image = self.scheduler.step(model_output, t, image, generator=generator).prev_sample

        samples, _ = image.chunk(2, dim=0)
        samples = self.vae.decode(samples / 0.18215).sample

        if output_type == "pil":
            samples = self.numpy_to_pil(samples)

        if not return_dict:
            return (samples,)

        return ImagePipelineOutput(images=samples)

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.dit(combined, t, y)
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)
