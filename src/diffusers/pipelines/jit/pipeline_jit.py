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

from typing import Dict, List, Optional, Tuple, Union

import torch

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
        id2label: Optional[Dict[int, str]] = None,
    ):
        super().__init__()
        self.register_modules(transformer=transformer, scheduler=scheduler)

        self.labels = {}
        if id2label is not None:
            for key, value in id2label.items():
                for label in value.split(","):
                    self.labels[label.lstrip().rstrip()] = int(key)
            self.labels = dict(sorted(self.labels.items()))

    def get_label_ids(self, label: Union[str, List[str]]) -> List[int]:
        r"""
        Map label strings from ImageNet to corresponding class ids.
        """
        if not isinstance(label, list):
            label = list(label)

        for l in label:
            if l not in self.labels:
                raise ValueError(
                    f"{l} does not exist. Please make sure to select one of the following labels: \n {self.labels}."
                )
        return [self.labels[l] for l in label]

    @torch.no_grad()
    def __call__(
        self,
        class_labels: List[int],
        guidance_scale: float = 4.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 50,
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
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`ImagePipelineOutput`] instead of a plain tuple.
        """

        batch_size = len(class_labels)
        image_size = self.transformer.config.sample_size
        channels = self.transformer.config.in_channels

        # Prepare latents (pixel space noise)
        latents = randn_tensor(
            shape=(batch_size, channels, image_size, image_size),
            generator=generator,
            device=self._execution_device,
            dtype=self.transformer.dtype,
        )

        # Prepare conditions
        class_labels = torch.tensor(class_labels, device=self._execution_device).reshape(-1)
        null_class_val = self.transformer.config.num_classes
        class_null = torch.tensor([null_class_val] * batch_size, device=self._execution_device)
        class_labels_input = torch.cat([class_labels, class_null], 0) if guidance_scale > 1 else class_labels


        latent_model_input = latents

        # Denoising loop
        self.scheduler.set_timesteps(num_inference_steps)

        for i, t in enumerate(self.progress_bar(self.scheduler.timesteps)):
            # Expand latents for CFG if needed
            if guidance_scale > 1:
                model_input = torch.cat([latent_model_input] * 2)
            else:
                model_input = latent_model_input

            # Map scheduler sigma (1->0) to JiT timestep (0->1)
            sigma = self.scheduler.sigmas[i]
            jit_t = 1.0 - sigma

            # Prepare inputs for model
            timesteps_tensor = torch.tensor(
                [jit_t] * model_input.shape[0], device=self._execution_device, dtype=latent_model_input.dtype
            )

            # Predict x
            noise_pred_x = self.transformer(
                model_input, timestep=timesteps_tensor, class_labels=class_labels_input
            ).sample

            # Compute velocity v = (x - z) / (1 - t) = (x - z) / sigma
            sigma_clamped = max(sigma.item(), 1e-5)

            # The scheduler expects (z - x) / sigma to move towards x when integrating with negative dt
            v_pred_all = (model_input - noise_pred_x) / sigma_clamped

            if guidance_scale > 1:
                v_cond, v_uncond = v_pred_all.chunk(2, dim=0)
                model_output = v_uncond + guidance_scale * (v_cond - v_uncond)
            else:
                model_output = v_pred_all

            # Step
            latent_model_input = self.scheduler.step(model_output, t, latent_model_input).prev_sample

        samples = latent_model_input
        samples = (samples / 2 + 0.5).clamp(0, 1)
        samples = samples.cpu().permute(0, 2, 3, 1).float().numpy()

        if output_type == "pil":
            samples = self.numpy_to_pil(samples)

        if not return_dict:
            return (samples,)

        return ImagePipelineOutput(images=samples)
