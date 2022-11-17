# For training scripts and notebooks see https://github.com/teticio/audio-diffusion

from typing import Union

from ...models import AutoencoderKL, UNet2DConditionModel
from ...schedulers import DDIMScheduler, DDPMScheduler
from .pipeline_audio_diffusion import AudioDiffusionPipeline


class LatentAudioDiffusionPipeline(AudioDiffusionPipeline):
    def __init__(
        self, unet: UNet2DConditionModel, scheduler: Union[DDIMScheduler, DDPMScheduler], vqvae: AutoencoderKL
    ):
        super().__init__(unet=unet, scheduler=scheduler)
        self.register_modules(vqvae=vqvae)

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)
