# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import Union

from ...models import AutoencoderKL, Mel, UNet2DConditionModel
from ...schedulers import DDIMScheduler, DDPMScheduler
from .pipeline_audio_diffusion import AudioDiffusionPipeline


class LatentAudioDiffusionPipeline(AudioDiffusionPipeline):
    """
    This pipeline inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Parameters:
        vqae (AutoencoderKL): Variational AutoEncoder
        unet (UNet2DConditionModel): UNET model
        mel (Mel): transform audio <-> spectrogram
        scheduler (Scheduler): de-noising scheduler
    """

    def __init__(
        self,
        vqvae: AutoencoderKL,
        unet: UNet2DConditionModel,
        mel: Mel,
        scheduler: Union[DDIMScheduler, DDPMScheduler],
    ):
        super().__init__(unet=unet, mel=mel, scheduler=scheduler)
        self.register_modules(vqvae=vqvae)

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)
