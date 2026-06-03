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

    @torch.inference_mode()
    def __call__(
        self,
        class_labels: List[int],
        guidance_scale: float = 4.0,
        num_inference_steps: int = 50,
        noise_scale: float = 1.0,
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
        batch_size = len(class_labels)
        image_size = self.transformer.config.sample_size
        channels = self.transformer.config.in_channels

        # Pixel-space noise. The sample is kept in fp32 across the ODE integration and only the
        # transformer forward runs in `transformer.dtype` (same pattern as Flux/SD3): JiT predicts
        # clean `x` and the `(x - z) / (1 - t)` velocity is precision-sensitive, so accumulating the
        # sample in fp16/bf16 degrades into noise over many steps.
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
        sigmas = self.scheduler.sigmas.to(device=self._execution_device, dtype=torch.float32)

        for i, t in enumerate(self.progress_bar(self.scheduler.timesteps)):
            model_input = torch.cat([sample] * 2) if do_classifier_free_guidance else sample

            # diffusers FlowMatch sigma (1 -> 0) maps to the JiT timestep (0 -> 1)
            sigma = sigmas[i].to(torch.float32)
            timestep = (1.0 - sigma).to(self.transformer.dtype).reshape(1).expand(model_input.shape[0])

            x_pred = self.transformer(
                model_input.to(self.transformer.dtype), timestep=timestep, class_labels=class_labels_input
            ).sample

            # velocity in fp32; clamp the 1 / (1 - t) denominator at JiT's `t_eps` (0.05).
            # the scheduler integrates with a negative step, so we pass (z - x) / (1 - t).
            v = (model_input - x_pred.float()) / torch.clamp(sigma, min=0.05)

            if do_classifier_free_guidance:
                v_cond, v_uncond = v.chunk(2, dim=0)
                v = v_uncond + guidance_scale * (v_cond - v_uncond)

            sample = self.scheduler.step(v, t, sample).prev_sample

        samples = (sample / 2 + 0.5).clamp(0, 1)
        samples = samples.cpu().permute(0, 2, 3, 1).float().numpy()

        if output_type == "pil":
            samples = self.numpy_to_pil(samples)

        if not return_dict:
            return (samples,)

        return ImagePipelineOutput(images=samples)
