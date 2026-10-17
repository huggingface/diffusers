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


from typing import Callable

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from transformers import DINOv3ViTModel

from ...image_processor import PipelineImageInput, TripoSplatImageProcessor
from ...models import AutoencoderKLFlux2
from ...models.autoencoders.autoencoder_triposplat import TripoSplatGaussianDecoder
from ...models.transformers.transformer_triposplat import TripoSplatTransformer3DModel
from ...schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from ...utils import replace_example_docstring
from ...utils.torch_utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline
from .modeling_birefnet import BiRefNetModel
from .pipeline_output import TripoSplatPipelineOutput


EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from PIL import Image
        >>> from diffusers import TripoSplatPipeline
        >>> from diffusers.utils import export_to_gaussian_ply

        >>> dtype = {"default": torch.float16, "image_encoder": torch.bfloat16, "vae": torch.bfloat16}
        >>> pipe = TripoSplatPipeline.from_pretrained("./triposplat-diffusers", dtype=dtype).to("cuda")
        >>> image = Image.open("object.png")
        >>> output = pipe(image, num_gaussians=32768, generator=torch.Generator("cuda").manual_seed(42))
        >>> export_to_gaussian_ply(output.gaussians[0], "object.ply")
        ```
"""


class TripoSplatPipeline(DiffusionPipeline):
    """Generate 3D Gaussian splats from foreground images.

    Args:
        transformer (`TripoSplatTransformer3DModel`):
            Image-conditioned Gaussian and camera flow transformer.
        decoder (`TripoSplatGaussianDecoder`):
            Octree and Gaussian parameter decoder.
        image_encoder (`transformers.DINOv3ViTModel`):
            DINOv3 vision encoder.
        vae (`AutoencoderKLFlux2`):
            Image VAE used to produce packed conditioning latents.
        scheduler (`FlowMatchEulerDiscreteScheduler`):
            Euler flow scheduler with the reference sigma shift.
        background_remover (`BiRefNetModel`, *optional*):
            Foreground segmentation model for RGB inputs.
        canvas_size (`int`, defaults to `1024`):
            Side length of prepared input images.
    """

    model_cpu_offload_seq = "background_remover->image_encoder->vae->transformer->decoder"
    _optional_components = ["background_remover"]
    _callback_tensor_inputs = ["latents", "camera_latents", "encoder_hidden_states", "image_latents"]

    def __init__(
        self,
        transformer: TripoSplatTransformer3DModel,
        decoder: TripoSplatGaussianDecoder,
        image_encoder: DINOv3ViTModel,
        vae: AutoencoderKLFlux2,
        scheduler: FlowMatchEulerDiscreteScheduler,
        background_remover: BiRefNetModel | None = None,
        canvas_size: int = 1024,
    ) -> None:
        super().__init__()
        self.register_modules(
            transformer=transformer,
            decoder=decoder,
            image_encoder=image_encoder,
            vae=vae,
            scheduler=scheduler,
            background_remover=background_remover,
        )
        self.register_to_config(canvas_size=canvas_size)
        self.image_processor = TripoSplatImageProcessor(canvas_size=canvas_size)
        self.num_prefix_tokens = (
            self.image_encoder.config.num_register_tokens + 1 if getattr(self, "image_encoder", None) else 5
        )
        self.q_token_length = self.transformer.config.q_token_length if getattr(self, "transformer", None) else 8192
        self.latent_channels = self.transformer.config.in_channels if getattr(self, "transformer", None) else 16
        self.camera_channels = self.transformer.config.cam_channels if getattr(self, "transformer", None) else 5
        self.gaussians_per_point = self.decoder.config.gaussians_per_point if getattr(self, "decoder", None) else 32

    def check_inputs(
        self,
        num_images_per_prompt: int,
        num_gaussians: int | list[int],
        output_type: str,
        callback_on_step_end_tensor_inputs: list[str],
    ) -> None:
        """Validate output, batching, density, and callback arguments."""
        if output_type not in ("pt", "np", "latent"):
            raise ValueError("output_type must be 'pt', 'np', or 'latent'.")
        if not isinstance(num_images_per_prompt, int) or num_images_per_prompt < 1:
            raise ValueError("num_images_per_prompt must be a positive integer.")
        if not set(callback_on_step_end_tensor_inputs).issubset(self._callback_tensor_inputs):
            raise ValueError(f"Callback inputs must be drawn from {self._callback_tensor_inputs}.")
        counts = num_gaussians if isinstance(num_gaussians, list) else [num_gaussians]
        if not counts or any(not isinstance(count, int) or not 32768 <= count <= 262144 for count in counts):
            raise ValueError(
                "num_gaussians must be an integer or a nonempty list of integers between 32768 and 262144."
            )

    def prepare_image(
        self, image: PipelineImageInput, erode_radius: int = 1, is_preprocessed: bool = False
    ) -> list[Image.Image]:
        """Prepare foreground images, predicting alpha masks for RGB inputs when BiRefNet is supplied."""
        images = self.image_processor.to_pil(image)
        if not is_preprocessed:
            images = self.image_processor.resize_shortest_edge(images)
            for index, item in enumerate(images):
                if item.mode == "RGBA" and item.getchannel("A").getextrema()[0] < 255:
                    continue
                if self.background_remover is None:
                    raise ValueError(
                        "RGB inputs require background_remover, or set is_preprocessed=True for prepared RGB images."
                    )
                pixels = self.image_processor.preprocess([item.convert("RGB")])
                pixels = F.interpolate(
                    pixels, size=(self.background_remover.config.sample_size,) * 2, mode="bilinear", align_corners=True
                )
                mean = pixels.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
                std = pixels.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
                pixels = ((pixels - mean) / std).to(device=self._execution_device, dtype=self.background_remover.dtype)
                alpha = self.background_remover(pixels).sample
                alpha = F.interpolate(
                    alpha.float(), size=(item.height, item.width), mode="bilinear", align_corners=True
                )[0, 0]
                alpha = (alpha.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
                item = item.convert("RGB")
                item.putalpha(Image.fromarray(alpha))
                images[index] = item
        return self.image_processor.prepare_foreground(
            images, erode_radius=erode_radius, is_preprocessed=is_preprocessed
        )

    def encode_image(self, images: list[Image.Image], device: torch.device | str) -> torch.Tensor:
        """Encode prepared RGB images into normalized DINOv3 features."""
        pixels = self.image_processor.preprocess(images).to(device=device)
        mean = pixels.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = pixels.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        pixels = ((pixels - mean) / std).to(self.image_encoder.dtype)
        original_inv_freq = self.image_encoder.rope_embeddings.inv_freq
        try:
            self.image_encoder.rope_embeddings.inv_freq = original_inv_freq.to(dtype=self.image_encoder.dtype)
            features = self.image_encoder(pixel_values=pixels).last_hidden_state
        finally:
            self.image_encoder.rope_embeddings.inv_freq = original_inv_freq
        return F.layer_norm(features.float(), features.shape[-1:])

    def encode_vae_image(
        self,
        images: list[Image.Image],
        device: torch.device | str,
        generator: torch.Generator | list[torch.Generator] | None = None,
    ) -> torch.Tensor:
        """Sample and normalize packed VAE conditioning latents for each input image."""
        if isinstance(generator, list):
            if len(generator) % len(images):
                raise ValueError("The generator list must contain one generator per image or generated sample.")
            generator = generator[:: len(generator) // len(images)]
        pixels = self.image_processor.preprocess(images).to(device=device, dtype=self.vae.dtype)
        pixels = self.image_processor.normalize(pixels)
        parameters = self.vae.encode(pixels).latent_dist.parameters
        # TripoSplat samples the posterior variance without the distribution's log-variance clamp.
        mean, logvar = parameters.chunk(2, dim=1)
        noise = randn_tensor(mean.shape, generator=generator, device=device, dtype=mean.dtype)
        latents = mean + torch.exp(0.5 * logvar) * noise
        batch, channels, height, width = latents.shape
        latents = latents.view(batch, channels, height // 2, 2, width // 2, 2).permute(0, 1, 3, 5, 2, 4)
        latents = latents.reshape(batch, channels * 4, height // 2, width // 2)
        bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
        bn_std = torch.sqrt(self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.bn.eps).to(
            latents.device, latents.dtype
        )
        latents = ((latents - bn_mean) / bn_std).float().flatten(2).transpose(1, 2).contiguous()
        return F.pad(latents, (0, 0, self.num_prefix_tokens, 0))

    def prepare_latents(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device | str,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        camera_latents: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare Gaussian and camera noise for the denoising loop."""
        shape = (batch_size, self.q_token_length, self.latent_channels)
        camera_shape = (batch_size, 1, self.camera_channels)
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        elif tuple(latents.shape) != shape:
            raise ValueError(f"Expected latents with shape {shape}, got {tuple(latents.shape)}.")
        if camera_latents is None:
            camera_latents = randn_tensor(camera_shape, generator=generator, device=device, dtype=dtype)
        elif tuple(camera_latents.shape) != camera_shape:
            raise ValueError(f"Expected camera_latents with shape {camera_shape}, got {tuple(camera_latents.shape)}.")
        return latents.to(device=device, dtype=dtype), camera_latents.to(device=device, dtype=dtype)

    @property
    def num_timesteps(self) -> int:
        return self._num_timesteps

    @property
    def guidance_scale(self) -> float:
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self) -> bool:
        return self.guidance_scale > 1

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        image: PipelineImageInput,
        num_inference_steps: int = 20,
        guidance_scale: float = 3.0,
        num_gaussians: int | list[int] = 262144,
        num_images_per_prompt: int = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        decoder_generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        camera_latents: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        image_latents: torch.Tensor | None = None,
        erode_radius: int = 1,
        is_preprocessed: bool = False,
        output_type: str = "pt",
        return_dict: bool = True,
        callback_on_step_end: Callable | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],
    ) -> TripoSplatPipelineOutput | tuple:
        """
        Args:
            image (`PipelineImageInput`):
                Input image or batch. Tensors use channel-first layout; arrays use channel-last layout.
            num_inference_steps (`int`, defaults to `20`):
                Number of Euler denoising steps.
            guidance_scale (`float`, defaults to `3.0`):
                Classifier-free guidance scale. Values at or below one use only the conditional prediction.
            num_gaussians (`int` or `list[int]`, defaults to `262144`):
                Gaussian counts between 32768 and 262144, rounded to multiples of the decoder's Gaussians per point.
            num_images_per_prompt (`int`, defaults to `1`):
                Number of generated objects per input image.
            generator (`torch.Generator` or `list[torch.Generator]`, *optional*):
                Generator or one generator per generated sample for VAE sampling, initial noise, and decoding.
            decoder_generator (`torch.Generator` or `list[torch.Generator]`, *optional*):
                Override for octree sampling and point jitter. Uses `generator` when omitted.
            latents (`torch.Tensor`, *optional*):
                Initial Gaussian noise. Generated when omitted.
            camera_latents (`torch.Tensor`, *optional*):
                Initial camera noise. Generated when omitted.
            encoder_hidden_states (`torch.Tensor`, *optional*):
                Precomputed DINOv3 features, before expansion by `num_images_per_prompt`.
            image_latents (`torch.Tensor`, *optional*):
                Precomputed packed VAE features, before expansion by `num_images_per_prompt`.
            erode_radius (`int`, defaults to `1`):
                Foreground mask erosion radius.
            is_preprocessed (`bool`, defaults to `False`):
                Whether inputs are already cropped RGB images composited on black.
            output_type (`str`, defaults to `"pt"`):
                Gaussian output format: `"pt"` or `"np"`. Use `"latent"` to skip decoding.
            return_dict (`bool`, defaults to `True`):
                Whether to return a structured output or a tuple.
            callback_on_step_end (`Callable`, *optional*):
                Called with the pipeline, step index, timestep, and requested tensors after each denoising step.
            callback_on_step_end_tensor_inputs (`list[str]`, defaults to `["latents"]`):
                Tensor names to pass to the callback.

        Returns:
            `TripoSplatPipelineOutput` or `tuple`: Gaussian parameters, denoised latents, camera latents, and prepared
            input images. A list of densities returns one batched Gaussian tensor per density.

        Examples:
        """
        self.check_inputs(num_images_per_prompt, num_gaussians, output_type, callback_on_step_end_tensor_inputs)
        counts = num_gaussians if isinstance(num_gaussians, list) else [num_gaussians]
        counts = [round(count / self.gaussians_per_point) * self.gaussians_per_point for count in counts]
        self._guidance_scale = guidance_scale
        device = self._execution_device
        images = self.prepare_image(image, erode_radius, is_preprocessed)
        batch_size = len(images) * num_images_per_prompt
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError("Pass one generator per generated sample.")
        if encoder_hidden_states is None:
            encoder_hidden_states = self.encode_image(images, device)
        if image_latents is None:
            image_latents = self.encode_vae_image(images, device, generator)
        if encoder_hidden_states.shape[:2] != image_latents.shape[:2] or encoder_hidden_states.shape[0] != len(images):
            raise ValueError("The DINO and VAE features must have matching batch and token dimensions.")
        encoder_hidden_states = encoder_hidden_states.to(device).repeat_interleave(num_images_per_prompt, dim=0)
        image_latents = image_latents.to(device).repeat_interleave(num_images_per_prompt, dim=0)
        latents, camera_latents = self.prepare_latents(
            batch_size, torch.float32, device, generator, latents, camera_latents
        )
        sigmas = np.linspace(1.0, 0.0, num_inference_steps + 1)[:-1]
        self.scheduler.set_timesteps(sigmas=sigmas, device=device)
        self._num_timesteps = len(self.scheduler.timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for index, timestep in enumerate(self.scheduler.timesteps):
                noise_pred, camera_pred = self.transformer(
                    latents,
                    timestep.expand(batch_size),
                    encoder_hidden_states,
                    image_latents,
                    camera_latents,
                    return_dict=False,
                )
                if self.do_classifier_free_guidance:
                    negative_pred, negative_camera = self.transformer(
                        latents,
                        timestep.expand(batch_size),
                        torch.zeros_like(encoder_hidden_states),
                        torch.zeros_like(image_latents),
                        camera_latents,
                        return_dict=False,
                    )
                    noise_pred = self.guidance_scale * noise_pred - (self.guidance_scale - 1) * negative_pred
                    camera_pred = self.guidance_scale * camera_pred - (self.guidance_scale - 1) * negative_camera
                latent_size = latents[0].numel()
                prediction = torch.cat([noise_pred.flatten(1), camera_pred.flatten(1)], dim=1)
                sample = torch.cat([latents.flatten(1), camera_latents.flatten(1)], dim=1)
                sample = self.scheduler.step(prediction.float(), timestep, sample, return_dict=False)[0]
                latents = sample[:, :latent_size].reshape_as(latents)
                camera_latents = sample[:, latent_size:].reshape_as(camera_latents)
                if callback_on_step_end is not None:
                    tensors = {
                        "latents": latents,
                        "camera_latents": camera_latents,
                        "encoder_hidden_states": encoder_hidden_states,
                        "image_latents": image_latents,
                    }
                    updates = callback_on_step_end(
                        self, index, timestep, {name: tensors[name] for name in callback_on_step_end_tensor_inputs}
                    )
                    latents = updates.pop("latents", latents)
                    camera_latents = updates.pop("camera_latents", camera_latents)
                    encoder_hidden_states = updates.pop("encoder_hidden_states", encoder_hidden_states)
                    image_latents = updates.pop("image_latents", image_latents)
                progress_bar.update()
        gaussians = None
        if output_type != "latent":
            decoder_generator = generator if decoder_generator is None else decoder_generator
            decoded = [self.decoder(latents, count, generator=decoder_generator).sample for count in counts]
            if output_type == "np":
                decoded = [item.cpu().numpy() for item in decoded]
            gaussians = decoded if isinstance(num_gaussians, list) else decoded[0]
        self.maybe_free_model_hooks()
        output = TripoSplatPipelineOutput(
            gaussians=gaussians, latents=latents, camera_latents=camera_latents, preprocessed_images=images
        )
        return output if return_dict else output.to_tuple()
