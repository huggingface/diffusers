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


from typing import List, Optional, Tuple, Union

import torch

from ...utils import logging
from ..pipeline_utils import AudioPipelineOutput, DiffusionPipeline


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class DanceDiffusionPipeline(DiffusionPipeline):
    r"""
    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Parameters:
        unet ([`UNet1DModel`]): U-Net architecture to denoise the encoded image.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image. Can be one of
            [`IPNDMScheduler`].
    """

    def __init__(self, unet, scheduler):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        num_inference_steps: int = 100,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        audio_length_in_s: Optional[float] = None,
        return_dict: bool = True,
    ) -> Union[AudioPipelineOutput, Tuple]:
        r"""
        Args:
            batch_size (`int`, *optional*, defaults to 1):
                The number of audio samples to generate.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality audio sample at
                the expense of slower inference.
            generator (`torch.Generator`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            audio_length_in_s (`float`, *optional*, defaults to `self.unet.config.sample_size/self.unet.config.sample_rate`):
                The length of the generated audio sample in seconds. Note that the output of the pipeline, *i.e.*
                `sample_size`, will be `audio_length_in_s` * `self.unet.sample_rate`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.AudioPipelineOutput`] instead of a plain tuple.

        Returns:
            [`~pipelines.AudioPipelineOutput`] or `tuple`: [`~pipelines.utils.AudioPipelineOutput`] if `return_dict` is
            True, otherwise a `tuple. When returning a tuple, the first element is a list with the generated images.
        """

        if audio_length_in_s is None:
            audio_length_in_s = self.unet.config.sample_size / self.unet.config.sample_rate

        sample_size = audio_length_in_s * self.unet.sample_rate

        down_scale_factor = 2 ** len(self.unet.up_blocks)
        if sample_size < 3 * down_scale_factor:
            raise ValueError(
                f"{audio_length_in_s} is too small. Make sure it's bigger or equal to"
                f" {3 * down_scale_factor / self.unet.sample_rate}."
            )

        original_sample_size = int(sample_size)
        if sample_size % down_scale_factor != 0:
            sample_size = ((audio_length_in_s * self.unet.sample_rate) // down_scale_factor + 1) * down_scale_factor
            logger.info(
                f"{audio_length_in_s} is increased to {sample_size / self.unet.sample_rate} so that it can be handled"
                f" by the model. It will be cut to {original_sample_size / self.unet.sample_rate} after the denoising"
                " process."
            )
        sample_size = int(sample_size)

        dtype = next(iter(self.unet.parameters())).dtype
        shape = (batch_size, self.unet.in_channels, sample_size)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        rand_device = "cpu" if self.device.type == "mps" else self.device
        if isinstance(generator, list):
            shape = (1,) + shape[1:]
            audio = [
                torch.randn(shape, generator=generator[i], device=rand_device, dtype=self.unet.dtype)
                for i in range(batch_size)
            ]
            audio = torch.cat(audio, dim=0).to(self.device)
        else:
            audio = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype).to(self.device)

        # set step values
        self.scheduler.set_timesteps(num_inference_steps, device=audio.device)
        self.scheduler.timesteps = self.scheduler.timesteps.to(dtype)

        for t in self.progress_bar(self.scheduler.timesteps):
            # 1. predict noise model_output
            model_output = self.unet(audio, t).sample

            # 2. compute previous image: x_t -> t_t-1
            audio = self.scheduler.step(model_output, t, audio).prev_sample

        audio = audio.clamp(-1, 1).float().cpu().numpy()

        audio = audio[:, :, :original_sample_size]

        if not return_dict:
            return (audio,)

        return AudioPipelineOutput(audios=audio)
