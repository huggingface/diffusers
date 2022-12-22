from typing import List, Optional, Tuple, Union

import torch

from ...configuration_utils import FrozenDict
from ...models import AutoencoderKL, DiT
from ...pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from ...schedulers import DDPMScheduler
from ...utils import deprecate


class DiTPipeline(DiffusionPipeline):
    def __init__(self, dit: DiT, vae: AutoencoderKL, scheduler: DDPMScheduler):
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

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.progress_bar(self.scheduler.timesteps):
            # 1. predict noise model_output
            model_output = self.dit.forward_with_cfg(image, t, y, cfg_scale)

            # 2. compute previous image: x_t -> x_t-1
            image = self.scheduler.step(model_output, t, image, generator=generator).prev_sample

        samples, _ = image.chunk(2, dim=0)
        samples = self.vae.decode(samples / 0.18215).sample

        if output_type == "pil":
            samples = self.numpy_to_pil(samples)

        if not return_dict:
            return (samples,)

        return ImagePipelineOutput(images=samples)
