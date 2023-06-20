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

from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import PIL
import torch
from transformers import CLIPTextModelWithProjection, CLIPTokenizer

from ...models import PriorTransformer
from ...pipelines import DiffusionPipeline
from ...schedulers import HeunDiscreteScheduler
from ...utils import (
    BaseOutput,
    is_accelerate_available,
    logging,
    randn_tensor,
    replace_example_docstring,
)
from .camera import create_pan_cameras
from .params_proj import ShapEParamsProjModel
from .renderer import (
    BoundingBoxVolume,
    ImportanceRaySampler,
    MLPNeRSTFModel,
    StratifiedRaySampler,
    VoidNeRFModel,
)


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py

        ```
"""


def merge_results(self, a: [torch.Tensor], b: torch.Tensor, dim: int, indices: torch.Tensor):
    """
    :param a: [..., n_a, ...]. The other dictionary containing the b's may
        contain extra tensors from earlier calculations, so a can be None.
    :param b: [..., n_b, ...] :param dim: dimension to merge :param indices: how the merged results should be sorted at
    the end :return: a concatted and sorted tensor of size [..., n_a + n_b, ...]
    """
    merged = torch.cat([a, b], dim=dim)
    return torch.gather(merged, dim=dim, index=torch.broadcast_to(indices, merged.shape))


def integrate_samples(volume_range, ts, density, channels):
    r"""
    Function integrating the model output.

    Args:
        volume_range: Specifies the integral range [t0, t1]
        ts: timesteps
        density: torch.Tensor [batch_size, *shape, n_samples, 1]
        channels: torch.Tensor [batch_size, *shape, n_samples, n_channels]
    returns:
        channels: integrated rgb output weights: torch.Tensor [batch_size, *shape, n_samples, 1] (density
        *transmittance)[i] weight for each rgb output at [..., i, :]. transmittance: transmittance of this volume
    )
    """

    # 1. Calculate the weights
    _, _, dt = volume_range.partition(ts)
    ddensity = density * dt

    mass = torch.cumsum(ddensity, dim=-2)
    transmittance = torch.exp(-mass[..., -1, :])

    alphas = 1.0 - torch.exp(-ddensity)
    Ts = torch.exp(torch.cat([torch.zeros_like(mass[..., :1, :]), -mass[..., :-1, :]], dim=-2))
    # This is the probability of light hitting and reflecting off of
    # something at depth [..., i, :].
    weights = alphas * Ts

    # 2. Integrate channels
    channels = torch.sum(channels * weights, dim=-2)

    return channels, weights, transmittance


@dataclass
class ShapEPipelineOutput(BaseOutput):
    """
    Output class for ShapEPipeline.

    Args:
        images (`torch.FloatTensor`)
            a list of images for 3D rendering
    """

    images: Union[PIL.Image.Image, np.ndarray]


class ShapEPipeline(DiffusionPipeline):
    """
    Pipeline for generating latent representation of a 3D asset with Shap.E

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        prior ([`PriorTransformer`]):
            The canonincal unCLIP prior to approximate the image embedding from the text embedding.
        text_encoder ([`CLIPTextModelWithProjection`]):
            Frozen text-encoder.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        scheduler ([`HeunDiscreteScheduler`]):
            A scheduler to be used in combination with `prior` to generate image embedding.
    """

    def __init__(
        self,
        prior: PriorTransformer,
        text_encoder: CLIPTextModelWithProjection,
        tokenizer: CLIPTokenizer,
        scheduler: HeunDiscreteScheduler,
        params_proj: ShapEParamsProjModel,
        renderer: MLPNeRSTFModel,
    ):
        super().__init__()

        self.register_modules(
            prior=prior,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler,
            params_proj=params_proj,
            renderer=renderer,
        )
        self.void = VoidNeRFModel(background=[0.0, 0.0, 0.0], channel_scale=255.0)
        self.volume = BoundingBoxVolume(bbox_max=[1.0, 1.0, 1.0], bbox_min=[-1.0, -1.0, -1.0])

    def prepare_latents(self, shape, dtype, device, generator, latents, scheduler):
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents.to(device)
        latents = latents * scheduler.init_noise_sigma
        return latents

    def enable_sequential_cpu_offload(self, gpu_id=0):
        r"""
        Offloads all models to CPU using accelerate, significantly reducing memory usage. When called, the pipeline's
        models have their state dicts saved to CPU and then are moved to a `torch.device('meta') and loaded to GPU only
        when their specific submodule has its `forward` method called.
        """
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")

        models = [
            self.text_encoder,
        ]
        for cpu_offloaded_model in models:
            if cpu_offloaded_model is not None:
                cpu_offload(cpu_offloaded_model, device)

    @property
    def _execution_device(self):
        r"""
        Returns the device on which the pipeline's models will be executed. After calling
        `pipeline.enable_sequential_cpu_offload()` the execution device can only be inferred from Accelerate's module
        hooks.
        """
        if self.device != torch.device("meta") or not hasattr(self.text_encoder, "_hf_hook"):
            return self.device
        for module in self.text_encoder.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def _encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
    ):
        len(prompt) if isinstance(prompt, list) else 1

        # YiYi Notes: set pad_token_id to be 0, not sure why I can't set in the config file
        self.tokenizer.pad_token_id = 0
        # get prompt text embeddings
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            )

        text_encoder_output = self.text_encoder(text_input_ids.to(device))
        prompt_embeds = text_encoder_output.text_embeds

        prompt_embeds = prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)
        # in Shap-E it normalize the prompt_embeds and then later rescale it, not sure why
        # YiYi TO-DO: move rescale out of prior_transformer and apply it here
        prompt_embeds = prompt_embeds / torch.linalg.norm(prompt_embeds, dim=-1, keepdim=True)

        if do_classifier_free_guidance:
            negative_prompt_embeds = torch.zeros_like(prompt_embeds)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        return prompt_embeds

    @torch.no_grad()
    def render_rays(self, rays, sampler, n_samples, prev_model_out=None, render_with_direction=False):
        """
        Perform volumetric rendering over a partition of possible t's in the union of rendering volumes (written below
        with some abuse of notations)

            C(r) := sum(
                transmittance(t[i]) * integrate(
                    lambda t: density(t) * channels(t) * transmittance(t), [t[i], t[i + 1]],
                ) for i in range(len(parts))
            ) + transmittance(t[-1]) * void_model(t[-1]).channels

        where

        1) transmittance(s) := exp(-integrate(density, [t[0], s])) calculates the probability of light passing through
        the volume specified by [t[0], s]. (transmittance of 1 means light can pass freely) 2) density and channels are
        obtained by evaluating the appropriate part.model at time t. 3) [t[i], t[i + 1]] is defined as the range of t
        where the ray intersects (parts[i].volume \\ union(part.volume for part in parts[:i])) at the surface of the
        shell (if bounded). If the ray does not intersect, the integral over this segment is evaluated as 0 and
        transmittance(t[i + 1]) := transmittance(t[i]). 4) The last term is integration to infinity (e.g. [t[-1],
        math.inf]) that is evaluated by the void_model (i.e. we consider this space to be empty).

        args:
            rays: [batch_size x ... x 2 x 3] origin and direction. sampler: disjoint volume integrals. n_samples:
            number of ts to sample. prev_model_outputs: model outputs from the previous rendering step, including

        :return: A tuple of
            - `channels`
            - A importance samplers for additional fine-grained rendering
            - raw model output
        """
        origin, direction = rays[..., 0, :], rays[..., 1, :]

        # Integrate over [t[i], t[i + 1]]

        # 1 Intersect the rays with the current volume and sample ts to integrate along.
        vrange = self.volume.intersect(origin, direction, t0_lower=None)
        ts = sampler.sample(vrange.t0, vrange.t1, n_samples)
        ts = ts.to(rays.dtype)

        if prev_model_out is not None:
            # Append the previous ts now before fprop because previous
            # rendering used a different model and we can't reuse the output.
            ts = torch.sort(torch.cat([ts, prev_model_out.ts], dim=-2), dim=-2).values

        batch_size, *_shape, _t0_dim = vrange.t0.shape
        _, *ts_shape, _ts_dim = ts.shape

        # 2. Get the points along the ray and query the model
        directions = torch.broadcast_to(direction.unsqueeze(-2), [batch_size, *ts_shape, 3])
        positions = origin.unsqueeze(-2) + ts * directions

        optional_directions = directions if render_with_direction else None

        model_out = self.renderer(
            position=positions,
            direction=optional_directions,
            ts=ts,
            nerf_level="coarse" if prev_model_out is None else "fine",
        )

        # 3. Integrate the model results
        channels, weights, transmittance = integrate_samples(
            vrange, model_out.ts, model_out.density, model_out.channels
        )

        # 4. Clean up results that do not intersect with the volume.
        transmittance = torch.where(vrange.intersected, transmittance, torch.ones_like(transmittance))
        channels = torch.where(vrange.intersected, channels, torch.zeros_like(channels))
        # 5. integration to infinity (e.g. [t[-1], math.inf]) that is evaluated by the void_model (i.e. we consider this space to be empty).
        channels = channels + transmittance * self.void(origin)

        weighted_sampler = ImportanceRaySampler(vrange, ts=model_out.ts, weights=weights)

        return channels, weighted_sampler, model_out

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]],
        num_images_per_prompt: int = 1,
        num_inference_steps: int = 25,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        guidance_scale: float = 4.0,
        sigma_min: float = 1e-3,
        sigma_max: float = 160.0,
        size: int = 64,
        ray_batch_size: int = 4096,
        n_coarse_samples=64,
        n_fine_samples=128,
        output_type: Optional[str] = "pt",  # pt only
        return_dict: bool = True,
    ):
        """
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            num_inference_steps (`int`, *optional*, defaults to 100):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            guidance_scale (`float`, *optional*, defaults to 4.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            output_type (`str`, *optional*, defaults to `"pt"`):
                The output format of the generate image. Choose between: `"np"` (`np.array`) or `"pt"`
                (`torch.Tensor`).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.

        Examples:

        Returns:
            [`ShapEPipelineOutput`] or `tuple`
        """

        if isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        device = self._execution_device

        batch_size = batch_size * num_images_per_prompt

        do_classifier_free_guidance = guidance_scale > 1.0
        prompt_embeds = self._encode_prompt(prompt, device, num_images_per_prompt, do_classifier_free_guidance)

        # prior

        self.scheduler.set_timesteps(
            num_inference_steps, device=device, sigma_min=sigma_min, sigma_max=sigma_max, use_karras_sigmas=True
        )
        timesteps = self.scheduler.timesteps

        num_embeddings = self.prior.config.num_embeddings
        embedding_dim = self.prior.config.embedding_dim

        latents = self.prepare_latents(
            (batch_size, num_embeddings * embedding_dim),
            prompt_embeds.dtype,
            device,
            generator,
            latents,
            self.scheduler,
        )

        # YiYi notes: for testing only to match ldm, we can directly create a latents with desired shape: batch_size, num_embeddings, embedding_dim
        latents = latents.reshape(latents.shape[0], num_embeddings, embedding_dim)

        for i, t in enumerate(self.progress_bar(timesteps)):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            scaled_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            noise_pred = self.prior(
                scaled_model_input,
                timestep=t,
                proj_embedding=prompt_embeds,
            ).predicted_image_embedding

            # remove the variance
            noise_pred, _ = noise_pred.split(
                scaled_model_input.shape[2], dim=2
            )  # batch_size, num_embeddings, embedding_dim

            # clip between -1 and 1
            noise_pred = noise_pred.clamp(-1, 1)

            if do_classifier_free_guidance is not None:
                noise_pred_uncond, noise_pred = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred - noise_pred_uncond)

            latents = self.scheduler.step(
                noise_pred,
                timestep=t,
                sample=latents,
                step_index=i,
            ).prev_sample

        # project the the paramters from the generated latents
        projected_params = self.params_proj(latents)

        # update the mlp layers of the renderer
        for name, param in self.renderer.state_dict().items():
            if f"nerstf.{name}" in projected_params.keys():
                param.copy_(projected_params[f"nerstf.{name}"].squeeze(0))

        # create cameras object
        camera = create_pan_cameras(size)
        rays = camera.camera_rays
        rays = rays.to(device)
        n_batches = rays.shape[1] // ray_batch_size

        coarse_sampler = StratifiedRaySampler()

        images = []
        with self.progress_bar(total=n_batches) as progress_bar:
            for idx in range(n_batches):
                rays_batch = rays[:, idx * ray_batch_size : (idx + 1) * ray_batch_size]

                # render rays with coarse, stratified samples.
                _, fine_sampler, coarse_model_out = self.render_rays(rays_batch, coarse_sampler, n_coarse_samples)
                # Then, render with additional importance-weighted ray samples.
                channels, _, _ = self.render_rays(
                    rays_batch, fine_sampler, n_fine_samples, prev_model_out=coarse_model_out
                )

                images.append(channels)
                progress_bar.update()

        images = torch.cat(images, dim=1)
        images = images.view(*camera.shape, camera.height, camera.width, -1).squeeze(0)

        if output_type not in ["np", "pil"]:
            raise ValueError(f"Only the output types `pil` and `np` are supported not output_type={output_type}")

        images = images.cpu().numpy()

        if output_type == "pil":
            images = self.numpy_to_pil(images)

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (images,)

        return ShapEPipelineOutput(images=images)
