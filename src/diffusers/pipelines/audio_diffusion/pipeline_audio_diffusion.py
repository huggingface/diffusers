# Copyright 2023 The HuggingFace Team. All rights reserved.
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


from math import acos, sin
from typing import List, Tuple, Union

import numpy as np
import torch
from PIL import Image

from ...models import AutoencoderKL, UNet2DConditionModel
from ...schedulers import DDIMScheduler, DDPMScheduler
from ...utils import randn_tensor
from ..pipeline_utils import AudioPipelineOutput, BaseOutput, DiffusionPipeline, ImagePipelineOutput
from .mel import Mel


class AudioDiffusionPipeline(DiffusionPipeline):
    """
    Pipeline for audio diffusion.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Parameters:
        vqae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        unet ([`UNet2DConditionModel`]):
            A `UNet2DConditionModel` to denoise the encoded image latents.
        mel ([`Mel`]):
            Transform audio into a spectrogram.
        scheduler ([`DDIMScheduler`] or [`DDPMScheduler`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`] or [`DDPMScheduler`].
    """

    _optional_components = ["vqvae"]

    def __init__(
        self,
        vqvae: AutoencoderKL,
        unet: UNet2DConditionModel,
        mel: Mel,
        scheduler: Union[DDIMScheduler, DDPMScheduler],
    ):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler, mel=mel, vqvae=vqvae)

    def get_default_steps(self) -> int:
        """Returns default number of steps recommended for inference.

        Returns:
            `int`:
                The number of steps.
        """
        return 50 if isinstance(self.scheduler, DDIMScheduler) else 1000

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        audio_file: str = None,
        raw_audio: np.ndarray = None,
        slice: int = 0,
        start_step: int = 0,
        steps: int = None,
        generator: torch.Generator = None,
        mask_start_secs: float = 0,
        mask_end_secs: float = 0,
        step_generator: torch.Generator = None,
        eta: float = 0,
        noise: torch.Tensor = None,
        encoding: torch.Tensor = None,
        return_dict=True,
    ) -> Union[
        Union[AudioPipelineOutput, ImagePipelineOutput],
        Tuple[List[Image.Image], Tuple[int, List[np.ndarray]]],
    ]:
        """
        The call function to the pipeline for generation.

        Args:
            batch_size (`int`):
                Number of samples to generate.
            audio_file (`str`):
                An audio file that must be on disk due to [Librosa](https://librosa.org/) limitation.
            raw_audio (`np.ndarray`):
                The raw audio file as a NumPy array.
            slice (`int`):
                Slice number of audio to convert.
            start_step (int):
                Step to start diffusion from.
            steps (`int`):
                Number of denoising steps (defaults to `50` for DDIM and `1000` for DDPM).
            generator (`torch.Generator`):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            mask_start_secs (`float`):
                Number of seconds of audio to mask (not generate) at start.
            mask_end_secs (`float`):
                Number of seconds of audio to mask (not generate) at end.
            step_generator (`torch.Generator`):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) used to denoise.
                None
            eta (`float`):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            noise (`torch.Tensor`):
                A noise tensor of shape `(batch_size, 1, height, width)` or `None`.
            encoding (`torch.Tensor`):
                A tensor for [`UNet2DConditionModel`] of shape `(batch_size, seq_length, cross_attention_dim)`.
            return_dict (`bool`):
                Whether or not to return a [`AudioPipelineOutput`], [`ImagePipelineOutput`] or a plain tuple.

        Examples:

        For audio diffusion:

        ```py
        import torch
        from IPython.display import Audio
        from diffusers import DiffusionPipeline

        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = DiffusionPipeline.from_pretrained("teticio/audio-diffusion-256").to(device)

        output = pipe()
        display(output.images[0])
        display(Audio(output.audios[0], rate=mel.get_sample_rate()))
        ```

        For latent audio diffusion:

        ```py
        import torch
        from IPython.display import Audio
        from diffusers import DiffusionPipeline

        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = DiffusionPipeline.from_pretrained("teticio/latent-audio-diffusion-256").to(device)

        output = pipe()
        display(output.images[0])
        display(Audio(output.audios[0], rate=pipe.mel.get_sample_rate()))
        ```

        For other tasks like variation, inpainting, outpainting, etc:

        ```py
        output = pipe(
            raw_audio=output.audios[0, 0],
            start_step=int(pipe.get_default_steps() / 2),
            mask_start_secs=1,
            mask_end_secs=1,
        )
        display(output.images[0])
        display(Audio(output.audios[0], rate=pipe.mel.get_sample_rate()))
        ```

        Returns:
            `List[PIL Image]`:
                A list of Mel spectrograms (`float`, `List[np.ndarray]`) with the sample rate and raw audio.
        """

        steps = steps or self.get_default_steps()
        self.scheduler.set_timesteps(steps)
        step_generator = step_generator or generator
        # For backwards compatibility
        if type(self.unet.config.sample_size) == int:
            self.unet.config.sample_size = (self.unet.config.sample_size, self.unet.config.sample_size)
        if noise is None:
            noise = randn_tensor(
                (
                    batch_size,
                    self.unet.config.in_channels,
                    self.unet.config.sample_size[0],
                    self.unet.config.sample_size[1],
                ),
                generator=generator,
                device=self.device,
            )
        images = noise
        mask = None

        if audio_file is not None or raw_audio is not None:
            self.mel.load_audio(audio_file, raw_audio)
            input_image = self.mel.audio_slice_to_image(slice)
            input_image = np.frombuffer(input_image.tobytes(), dtype="uint8").reshape(
                (input_image.height, input_image.width)
            )
            input_image = (input_image / 255) * 2 - 1
            input_images = torch.tensor(input_image[np.newaxis, :, :], dtype=torch.float).to(self.device)

            if self.vqvae is not None:
                input_images = self.vqvae.encode(torch.unsqueeze(input_images, 0)).latent_dist.sample(
                    generator=generator
                )[0]
                input_images = self.vqvae.config.scaling_factor * input_images

            if start_step > 0:
                images[0, 0] = self.scheduler.add_noise(input_images, noise, self.scheduler.timesteps[start_step - 1])

            pixels_per_second = (
                self.unet.config.sample_size[1] * self.mel.get_sample_rate() / self.mel.x_res / self.mel.hop_length
            )
            mask_start = int(mask_start_secs * pixels_per_second)
            mask_end = int(mask_end_secs * pixels_per_second)
            mask = self.scheduler.add_noise(input_images, noise, torch.tensor(self.scheduler.timesteps[start_step:]))

        for step, t in enumerate(self.progress_bar(self.scheduler.timesteps[start_step:])):
            if isinstance(self.unet, UNet2DConditionModel):
                model_output = self.unet(images, t, encoding)["sample"]
            else:
                model_output = self.unet(images, t)["sample"]

            if isinstance(self.scheduler, DDIMScheduler):
                images = self.scheduler.step(
                    model_output=model_output,
                    timestep=t,
                    sample=images,
                    eta=eta,
                    generator=step_generator,
                )["prev_sample"]
            else:
                images = self.scheduler.step(
                    model_output=model_output,
                    timestep=t,
                    sample=images,
                    generator=step_generator,
                )["prev_sample"]

            if mask is not None:
                if mask_start > 0:
                    images[:, :, :, :mask_start] = mask[:, step, :, :mask_start]
                if mask_end > 0:
                    images[:, :, :, -mask_end:] = mask[:, step, :, -mask_end:]

        if self.vqvae is not None:
            # 0.18215 was scaling factor used in training to ensure unit variance
            images = 1 / self.vqvae.config.scaling_factor * images
            images = self.vqvae.decode(images)["sample"]

        images = (images / 2 + 0.5).clamp(0, 1)
        images = images.cpu().permute(0, 2, 3, 1).numpy()
        images = (images * 255).round().astype("uint8")
        images = list(
            (Image.fromarray(_[:, :, 0]) for _ in images)
            if images.shape[3] == 1
            else (Image.fromarray(_, mode="RGB").convert("L") for _ in images)
        )

        audios = [self.mel.image_to_audio(_) for _ in images]
        if not return_dict:
            return images, (self.mel.get_sample_rate(), audios)

        return BaseOutput(**AudioPipelineOutput(np.array(audios)[:, np.newaxis, :]), **ImagePipelineOutput(images))

    @torch.no_grad()
    def encode(self, images: List[Image.Image], steps: int = 50) -> np.ndarray:
        """
        Reverse the denoising step process to recover a noisy image from the generated image.

        Args:
            images (`List[PIL Image]`):
                List of images to encode.
            steps (`int`):
                Number of encoding steps to perform (defaults to `50`).

        Returns:
            `np.ndarray`:
                A noise tensor of shape `(batch_size, 1, height, width)`.
        """

        # Only works with DDIM as this method is deterministic
        assert isinstance(self.scheduler, DDIMScheduler)
        self.scheduler.set_timesteps(steps)
        sample = np.array(
            [np.frombuffer(image.tobytes(), dtype="uint8").reshape((1, image.height, image.width)) for image in images]
        )
        sample = (sample / 255) * 2 - 1
        sample = torch.Tensor(sample).to(self.device)

        for t in self.progress_bar(torch.flip(self.scheduler.timesteps, (0,))):
            prev_timestep = t - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
            alpha_prod_t = self.scheduler.alphas_cumprod[t]
            alpha_prod_t_prev = (
                self.scheduler.alphas_cumprod[prev_timestep]
                if prev_timestep >= 0
                else self.scheduler.final_alpha_cumprod
            )
            beta_prod_t = 1 - alpha_prod_t
            model_output = self.unet(sample, t)["sample"]
            pred_sample_direction = (1 - alpha_prod_t_prev) ** (0.5) * model_output
            sample = (sample - pred_sample_direction) * alpha_prod_t_prev ** (-0.5)
            sample = sample * alpha_prod_t ** (0.5) + beta_prod_t ** (0.5) * model_output

        return sample

    @staticmethod
    def slerp(x0: torch.Tensor, x1: torch.Tensor, alpha: float) -> torch.Tensor:
        """Spherical Linear intERPolation.

        Args:
            x0 (`torch.Tensor`):
                The first tensor to interpolate between.
            x1 (`torch.Tensor`):
                Second tensor to interpolate between.
            alpha (`float`):
                Interpolation between 0 and 1

        Returns:
            `torch.Tensor`:
                The interpolated tensor.
        """

        theta = acos(torch.dot(torch.flatten(x0), torch.flatten(x1)) / torch.norm(x0) / torch.norm(x1))
        return sin((1 - alpha) * theta) * x0 / sin(theta) + sin(alpha * theta) * x1 / sin(theta)
