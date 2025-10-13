# Copyright 2025 Alpha-VLLM and The HuggingFace Team. All rights reserved.
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

from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
from transformers import AutoModel, AutoTokenizer

from ...callbacks import MultiPipelineCallbacks, PipelineCallback
from ...image_processor import VaeImageProcessor
from ...models import AutoencoderKL
from ...models.transformers.transformer_lumina_dit import LuminaDiT2DModel
from ...schedulers import LuminaFlowMatchScheduler
from ...utils import (
    logging,
    replace_example_docstring,
)
from ...utils.torch_utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline, ImagePipelineOutput


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import LuminaT2IPipeline

        >>> pipe = LuminaT2IPipeline.from_pretrained(
        ...     "Alpha-VLLM/Lumina-T2I", torch_dtype=torch.bfloat16
        ... )
        >>> pipe = pipe.to("cuda")

        >>> # Enable memory optimizations.
        >>> pipe.enable_model_cpu_offload()

        >>> prompt = "A photo of a cat"
        >>> image = pipe(prompt, num_inference_steps=30, guidance_scale=4.0).images[0]
        >>> image.save("lumina_cat.png")
        ```
"""


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call.
    """
    if timesteps is not None:
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class LuminaT2IPipeline(DiffusionPipeline):
    r"""
    Pipeline for text-to-image generation using Lumina-T2I.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        transformer ([`LuminaDiT2DModel`]):
            The Lumina DiT model for denoising the encoded image latents.
        scheduler ([`LuminaFlowMatchScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        text_encoder ([`~transformers.LlamaModel`]):
            LLaMA text encoder for encoding prompts.
        tokenizer ([`~transformers.LlamaTokenizer`]):
            Tokenizer for the text encoder.
    """

    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _optional_components = []
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

    def __init__(
        self,
        transformer: LuminaDiT2DModel,
        scheduler: LuminaFlowMatchScheduler,
        vae: AutoencoderKL,
        text_encoder: AutoModel,
        tokenizer: AutoTokenizer,
    ):
        super().__init__()

        self.register_modules(
            transformer=transformer,
            scheduler=scheduler,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
        )

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        device: torch.device,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        max_sequence_length: int = 128,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            prompt_attention_mask (`torch.Tensor`, *optional*):
                Attention mask for the prompt.
            negative_prompt_attention_mask (`torch.Tensor`, *optional*):
                Attention mask for the negative prompt.
            max_sequence_length (`int`, defaults to 128):
                Maximum sequence length to use for text encoding.
        """
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            # Tokenize
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=max_sequence_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(untruncated_ids[:, max_sequence_length - 1 : -1])
                logger.warning(
                    f"The following part of your input was truncated because the model can only handle sequences of length {max_sequence_length}: {removed_text}"
                )

            prompt_attention_mask = text_inputs.attention_mask.to(device)
            
            # Encode with text encoder
            prompt_embeds = self.text_encoder(
                text_input_ids.to(device),
                attention_mask=prompt_attention_mask,
            ).last_hidden_state

        # Duplicate text embeddings for each generation per prompt
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1, prompt_embeds.shape[-1])
        
        if prompt_attention_mask is not None:
            prompt_attention_mask = prompt_attention_mask.repeat(1, num_images_per_prompt)
            prompt_attention_mask = prompt_attention_mask.view(batch_size * num_images_per_prompt, -1)

        # Get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_sequence_length,
                truncation=True,
                return_tensors="pt",
            )
            
            negative_prompt_attention_mask = uncond_input.attention_mask.to(device)
            
            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=negative_prompt_attention_mask,
            ).last_hidden_state

        if do_classifier_free_guidance:
            # Duplicate unconditional embeddings for each generation per prompt
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(
                batch_size * num_images_per_prompt, -1, negative_prompt_embeds.shape[-1]
            )
            
            if negative_prompt_attention_mask is not None:
                negative_prompt_attention_mask = negative_prompt_attention_mask.repeat(1, num_images_per_prompt)
                negative_prompt_attention_mask = negative_prompt_attention_mask.view(
                    batch_size * num_images_per_prompt, -1
                )

            # For classifier free guidance, we need to do two forward passes.
            # We concatenate the unconditional and conditional embeddings into a single batch
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            if prompt_attention_mask is not None and negative_prompt_attention_mask is not None:
                prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask])

        return prompt_embeds, prompt_attention_mask

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.Tensor] = None,
    ):
        shape = (
            batch_size,
            num_channels_latents,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # Scale the initial noise by the standard deviation required by the scheduler
        # (not needed for rectified flow)
        return latents

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1.0

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 30,
        timesteps: Optional[List[int]] = None,
        guidance_scale: float = 4.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 128,
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            height (`int`, defaults to 1024):
                The height in pixels of the generated image.
            width (`int`, defaults to 1024):
                The width in pixels of the generated image.
            num_inference_steps (`int`, defaults to 30):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process. If not defined, equal spaced `num_inference_steps`
                timesteps are used.
            guidance_scale (`float`, defaults to 4.0):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_images_per_prompt (`int`, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            prompt_attention_mask (`torch.Tensor`, *optional*):
                Attention mask for the prompt.
            negative_prompt_attention_mask (`torch.Tensor`, *optional*):
                Attention mask for the negative prompt.
            output_type (`str`, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising step during inference.
            callback_on_step_end_tensor_inputs (`List[str]`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function.
            max_sequence_length (`int`, defaults to 128):
                Maximum sequence length to use for text encoding.

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images.
        """
        # 0. Check inputs
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # 1. Define call parameters
        device = self._execution_device
        self._guidance_scale = guidance_scale
        self._interrupt = False

        # 2. Encode input prompt
        prompt_embeds, prompt_attention_mask = self.encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            max_sequence_length=max_sequence_length,
        )

        # 3. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        self._num_timesteps = len(timesteps)

        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 5. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # Expand latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents

                # Broadcast timestep to batch dimension
                timestep = t.expand(latent_model_input.shape[0])

                # Predict the noise residual
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    encoder_attention_mask=prompt_attention_mask,
                    return_dict=False,
                )[0]

                # Perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # Compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    
                    latents = callback_outputs.pop("latents", latents)

                progress_bar.update()

        # 6. Decode latents
        if output_type != "latent":
            # Adjust for VAE scaling
            vae_scale = 0.18215 if self.vae.config.scaling_factor is None else self.vae.config.scaling_factor
            latents = latents / vae_scale
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)
        else:
            image = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)

