# Copyright 2025 Microsoft and The HuggingFace Team. All rights reserved.
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

from typing import Callable, List, Optional, Tuple, Union

import torch
from transformers import CLIPTextModel, CLIPTokenizer

from ....configuration_utils import ConfigMixin, register_to_config
from ....models import ModelMixin, Transformer2DModel, VQModel
from ....schedulers import VQDiffusionScheduler
from ....utils import logging
from ...pipeline_utils import DiffusionPipeline, ImagePipelineOutput


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class LearnedClassifierFreeSamplingEmbeddings(ModelMixin, ConfigMixin):
    """
    Utility class for storing learned text embeddings for classifier free sampling
    """

    @register_to_config
    def __init__(self, learnable: bool, hidden_size: Optional[int] = None, length: Optional[int] = None):
        super().__init__()

        self.learnable = learnable

        if self.learnable:
            assert hidden_size is not None, "learnable=True requires `hidden_size` to be set"
            assert length is not None, "learnable=True requires `length` to be set"

            embeddings = torch.zeros(length, hidden_size)
        else:
            embeddings = None

        self.embeddings = torch.nn.Parameter(embeddings)


class VQDiffusionPipeline(DiffusionPipeline):
    r"""
    Pipeline for text-to-image generation using VQ Diffusion.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        vqvae ([`VQModel`]):
            Vector Quantized Variational Auto-Encoder (VAE) model to encode and decode images to and from latent
            representations.
        text_encoder ([`~transformers.CLIPTextModel`]):
            Frozen text-encoder ([clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32)).
        tokenizer ([`~transformers.CLIPTokenizer`]):
            A `CLIPTokenizer` to tokenize text.
        transformer ([`Transformer2DModel`]):
            A conditional `Transformer2DModel` to denoise the encoded image latents.
        scheduler ([`VQDiffusionScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
    """

    vqvae: VQModel
    text_encoder: CLIPTextModel
    tokenizer: CLIPTokenizer
    transformer: Transformer2DModel
    learned_classifier_free_sampling_embeddings: LearnedClassifierFreeSamplingEmbeddings
    scheduler: VQDiffusionScheduler

    def __init__(
        self,
        vqvae: VQModel,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        transformer: Transformer2DModel,
        scheduler: VQDiffusionScheduler,
        learned_classifier_free_sampling_embeddings: LearnedClassifierFreeSamplingEmbeddings,
    ):
        super().__init__()

        self.register_modules(
            vqvae=vqvae,
            transformer=transformer,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler,
            learned_classifier_free_sampling_embeddings=learned_classifier_free_sampling_embeddings,
        )

    def _encode_prompt(self, prompt, num_images_per_prompt, do_classifier_free_guidance):
        batch_size = len(prompt) if isinstance(prompt, list) else 1

        # get prompt text embeddings
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids

        if text_input_ids.shape[-1] > self.tokenizer.model_max_length:
            removed_text = self.tokenizer.batch_decode(text_input_ids[:, self.tokenizer.model_max_length :])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            )
            text_input_ids = text_input_ids[:, : self.tokenizer.model_max_length]
        prompt_embeds = self.text_encoder(text_input_ids.to(self.device))[0]

        # NOTE: This additional step of normalizing the text embeddings is from VQ-Diffusion.
        # While CLIP does normalize the pooled output of the text transformer when combining
        # the image and text embeddings, CLIP does not directly normalize the last hidden state.
        #
        # CLIP normalizing the pooled output.
        # https://github.com/huggingface/transformers/blob/d92e22d1f28324f513f3080e5c47c071a3916721/src/transformers/models/clip/modeling_clip.py#L1052-L1053
        prompt_embeds = prompt_embeds / prompt_embeds.norm(dim=-1, keepdim=True)

        # duplicate text embeddings for each generation per prompt
        prompt_embeds = prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)

        if do_classifier_free_guidance:
            if self.learned_classifier_free_sampling_embeddings.learnable:
                negative_prompt_embeds = self.learned_classifier_free_sampling_embeddings.embeddings
                negative_prompt_embeds = negative_prompt_embeds.unsqueeze(0).repeat(batch_size, 1, 1)
            else:
                uncond_tokens = [""] * batch_size

                max_length = text_input_ids.shape[-1]
                uncond_input = self.tokenizer(
                    uncond_tokens,
                    padding="max_length",
                    max_length=max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                negative_prompt_embeds = self.text_encoder(uncond_input.input_ids.to(self.device))[0]
                # See comment for normalizing text embeddings
                negative_prompt_embeds = negative_prompt_embeds / negative_prompt_embeds.norm(dim=-1, keepdim=True)

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        return prompt_embeds

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        num_inference_steps: int = 100,
        guidance_scale: float = 5.0,
        truncation_rate: float = 1.0,
        num_images_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        callback_steps: int = 1,
    ) -> Union[ImagePipelineOutput, Tuple]:
        """
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide image generation.
            num_inference_steps (`int`, *optional*, defaults to 100):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            truncation_rate (`float`, *optional*, defaults to 1.0 (equivalent to no truncation)):
                Used to "truncate" the predicted classes for x_0 such that the cumulative probability for a pixel is at
                most `truncation_rate`. The lowest probabilities that would increase the cumulative probability above
                `truncation_rate` are set to zero.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor` of shape (batch), *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Must be valid embedding indices.If not provided, a latents tensor will be generated of
                completely masked latent pixels.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.
            callback (`Callable`, *optional*):
                A function that calls every `callback_steps` steps during inference. The function is called with the
                following arguments: `callback(step: int, timestep: int, latents: torch.Tensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function is called. If not specified, the callback is called at
                every step.

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images.
        """
        if isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        batch_size = batch_size * num_images_per_prompt

        do_classifier_free_guidance = guidance_scale > 1.0

        prompt_embeds = self._encode_prompt(prompt, num_images_per_prompt, do_classifier_free_guidance)

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        # get the initial completely masked latents unless the user supplied it

        latents_shape = (batch_size, self.transformer.num_latent_pixels)
        if latents is None:
            mask_class = self.transformer.num_vector_embeds - 1
            latents = torch.full(latents_shape, mask_class).to(self.device)
        else:
            if latents.shape != latents_shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {latents_shape}")
            if (latents < 0).any() or (latents >= self.transformer.num_vector_embeds).any():
                raise ValueError(
                    "Unexpected latents value(s). All latents be valid embedding indices i.e. in the range 0,"
                    f" {self.transformer.num_vector_embeds - 1} (inclusive)."
                )
            latents = latents.to(self.device)

        # set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)

        timesteps_tensor = self.scheduler.timesteps.to(self.device)

        sample = latents

        for i, t in enumerate(self.progress_bar(timesteps_tensor)):
            # expand the sample if we are doing classifier free guidance
            latent_model_input = torch.cat([sample] * 2) if do_classifier_free_guidance else sample

            # predict the un-noised image
            # model_output == `log_p_x_0`
            model_output = self.transformer(latent_model_input, encoder_hidden_states=prompt_embeds, timestep=t).sample

            if do_classifier_free_guidance:
                model_output_uncond, model_output_text = model_output.chunk(2)
                model_output = model_output_uncond + guidance_scale * (model_output_text - model_output_uncond)
                model_output -= torch.logsumexp(model_output, dim=1, keepdim=True)

            model_output = self.truncate(model_output, truncation_rate)

            # remove `log(0)`'s (`-inf`s)
            model_output = model_output.clamp(-70)

            # compute the previous noisy sample x_t -> x_t-1
            sample = self.scheduler.step(model_output, timestep=t, sample=sample, generator=generator).prev_sample

            # call the callback, if provided
            if callback is not None and i % callback_steps == 0:
                callback(i, t, sample)

        embedding_channels = self.vqvae.config.vq_embed_dim
        embeddings_shape = (batch_size, self.transformer.height, self.transformer.width, embedding_channels)
        embeddings = self.vqvae.quantize.get_codebook_entry(sample, shape=embeddings_shape)
        image = self.vqvae.decode(embeddings, force_not_quantize=True).sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()

        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)

    def truncate(self, log_p_x_0: torch.Tensor, truncation_rate: float) -> torch.Tensor:
        """
        Truncates `log_p_x_0` such that for each column vector, the total cumulative probability is `truncation_rate`
        The lowest probabilities that would increase the cumulative probability above `truncation_rate` are set to
        zero.
        """
        sorted_log_p_x_0, indices = torch.sort(log_p_x_0, 1, descending=True)
        sorted_p_x_0 = torch.exp(sorted_log_p_x_0)
        keep_mask = sorted_p_x_0.cumsum(dim=1) < truncation_rate

        # Ensure that at least the largest probability is not zeroed out
        all_true = torch.full_like(keep_mask[:, 0:1, :], True)
        keep_mask = torch.cat((all_true, keep_mask), dim=1)
        keep_mask = keep_mask[:, :-1, :]

        keep_mask = keep_mask.gather(1, indices.argsort(1))

        rv = log_p_x_0.clone()

        rv[~keep_mask] = -torch.inf  # -inf = log(0)

        return rv
