# Copyright 2026 The HuggingFace Team. All rights reserved.
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

from ...image_processor import VaeImageProcessor
from ...models.transformers import JiTTransformer2DModel
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils.torch_utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline, ImagePipelineOutput


class JiTPipeline(DiffusionPipeline):
    r"""
    Pipeline for image generation using JiT (Just image Transformer).

    Parameters:
        transformer ([`JiTTransformer2DModel`]):
            A class conditioned `JiTTransformer2DModel` to denoise the images.
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the images.
    """

    model_cpu_offload_seq = "transformer"

    def __init__(
        self,
        transformer: JiTTransformer2DModel,
        scheduler: FlowMatchEulerDiscreteScheduler,
    ):
        super().__init__()
        self.register_modules(transformer=transformer, scheduler=scheduler)
        self.image_processor = VaeImageProcessor(vae_scale_factor=1)

    @torch.no_grad()
    def __call__(
        self,
        class_labels: Union[int, List[int]],
        guidance_scale: float = 4.0,
        num_inference_steps: int = 50,
        noise_scale: Optional[float] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        The call function to the pipeline for generation.

        Args:
            class_labels (List[int]):
                List of ImageNet class labels for the images to be generated.
            guidance_scale (`float`, *optional*, defaults to 4.0):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            generator (`torch.Generator`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps.
            noise_scale (`float`, *optional*, defaults to 1.0):
                Standard deviation of the initial pixel-space noise. JiT scales this with resolution to keep the
                signal-to-noise ratio constant (`1.0` at 256, `2.0` at 512, `4.0` at 1024).
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`ImagePipelineOutput`] instead of a plain tuple.
        """

        do_classifier_free_guidance = guidance_scale > 1.0
        if not isinstance(class_labels, list):
            class_labels = [class_labels]
            
        batch_size = len(class_labels)
        image_size = self.transformer.config.sample_size
        channels = self.transformer.config.in_channels

        noise_scale = noise_scale if noise_scale is not None else (image_size / 256.0)

        # Pixel-space noise.
        sample = noise_scale * randn_tensor(
            shape=(batch_size, channels, image_size, image_size),
            generator=generator,
            device=self._execution_device,
            dtype=torch.float32,
        )

        class_labels = torch.tensor(class_labels, device=self._execution_device).reshape(-1)
        null_class = torch.full_like(class_labels, self.transformer.config.num_classes)
        class_labels_input = torch.cat([class_labels, null_class], 0) if do_classifier_free_guidance else class_labels

        self.scheduler.set_timesteps(num_inference_steps, device=self._execution_device)
        timesteps = self.scheduler.timesteps
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                model_input = torch.cat([sample] * 2) if do_classifier_free_guidance else sample

                sigma = t / self.scheduler.config.num_train_timesteps
                sigma = sigma.to(torch.float32)
                timestep = (1.0 - sigma).to(self.transformer.dtype).expand(model_input.shape[0])

                x_pred = self.transformer(
                    model_input.to(self.transformer.dtype), timestep=timestep, class_labels=class_labels_input
                ).sample

                # velocity in fp32; clamp the 1 / (1 - t) denominator at JiT's `t_eps` (0.05).
                v = (model_input - x_pred.float()) / torch.clamp(sigma, min=0.05)

                if do_classifier_free_guidance:
                    v_cond, v_uncond = v.chunk(2, dim=0)
                    v = v_uncond + guidance_scale * (v_cond - v_uncond)

                sample = self.scheduler.step(v, t, sample).prev_sample

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        samples = (sample / 2 + 0.5).clamp(0, 1)

        if output_type in ["latent", "pt"]:
            samples = samples
        else:
            samples = self.image_processor.postprocess(samples, output_type=output_type)

        if not return_dict:
            return (samples,)

        return ImagePipelineOutput(images=samples)
