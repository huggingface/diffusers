# Copyright (c) Bria.ai. All rights reserved.
#
# This file is licensed under the Creative Commons Attribution-NonCommercial 4.0 International Public License (CC-BY-NC-4.0).
# You may obtain a copy of the license at https://creativecommons.org/licenses/by-nc/4.0/
#
# You are free to share and adapt this material for non-commercial purposes provided you give appropriate credit,
# indicate if changes were made, and do not use the material for commercial purposes.
#
# See the license for further details.

from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from transformers import AutoTokenizer
from transformers.models.smollm3.modeling_smollm3 import SmolLM3ForCausalLM

from ...image_processor import VaeImageProcessor
from ...loaders import FluxLoraLoaderMixin
from ...models.autoencoders.autoencoder_kl_wan import AutoencoderKLWan
from ...models.transformers.transformer_bria_fibo import BriaFiboTransformer2DModel
from ...pipelines.bria_fibo.pipeline_output import BriaFiboPipelineOutput
from ...pipelines.flux.pipeline_flux import calculate_shift, retrieve_timesteps
from ...pipelines.pipeline_utils import DiffusionPipeline
from ...schedulers import FlowMatchEulerDiscreteScheduler, KarrasDiffusionSchedulers
from ...utils import (
    USE_PEFT_BACKEND,
    is_torch_xla_available,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from ...utils.torch_utils import randn_tensor


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Example:
    ```python
    import torch
    from diffusers import BriaFiboPipeline
    from diffusers.modular_pipelines import ModularPipeline

    torch.set_grad_enabled(False)
    vlm_pipe = ModularPipeline.from_pretrained("briaai/FIBO-VLM-prompt-to-JSON", trust_remote_code=True)

    pipe = BriaFiboPipeline.from_pretrained(
        "briaai/FIBO",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    pipe.enable_model_cpu_offload()

    with torch.inference_mode():
        # 1. Create a prompt to generate an initial image
        output = vlm_pipe(prompt="a beautiful dog")
        json_prompt_generate = output.values["json_prompt"]

        # Generate the image from the structured json prompt
        results_generate = pipe(prompt=json_prompt_generate, num_inference_steps=50, guidance_scale=5)
        results_generate.images[0].save("image_generate.png")
    ```
"""


class BriaFiboPipeline(DiffusionPipeline):
    r"""
    Args:
        transformer (`BriaFiboTransformer2DModel`):
            The transformer model for 2D diffusion modeling.
        scheduler (`FlowMatchEulerDiscreteScheduler` or `KarrasDiffusionSchedulers`):
            Scheduler to be used with `transformer` to denoise the encoded latents.
        vae (`AutoencoderKLWan`):
            Variational Auto-Encoder for encoding and decoding images to and from latent representations.
        text_encoder (`SmolLM3ForCausalLM`):
            Text encoder for processing input prompts.
        tokenizer (`AutoTokenizer`):
            Tokenizer used for processing the input text prompts for the text_encoder.
    """

    model_cpu_offload_seq = "text_encoder->text_encoder_2->image_encoder->transformer->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds"]

    def __init__(
        self,
        transformer: BriaFiboTransformer2DModel,
        scheduler: Union[FlowMatchEulerDiscreteScheduler, KarrasDiffusionSchedulers],
        vae: AutoencoderKLWan,
        text_encoder: SmolLM3ForCausalLM,
        tokenizer: AutoTokenizer,
    ):
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            scheduler=scheduler,
        )

        self.vae_scale_factor = 16
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)
        self.default_sample_size = 64

    def get_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 2048,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if not prompt:
            raise ValueError("`prompt` must be a non-empty string or list of strings.")

        batch_size = len(prompt)
        bot_token_id = 128000

        text_encoder_device = device if device is not None else torch.device("cpu")
        if not isinstance(text_encoder_device, torch.device):
            text_encoder_device = torch.device(text_encoder_device)

        if all(p == "" for p in prompt):
            input_ids = torch.full((batch_size, 1), bot_token_id, dtype=torch.long, device=text_encoder_device)
            attention_mask = torch.ones_like(input_ids)
        else:
            tokenized = self.tokenizer(
                prompt,
                padding="longest",
                max_length=max_sequence_length,
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            input_ids = tokenized.input_ids.to(text_encoder_device)
            attention_mask = tokenized.attention_mask.to(text_encoder_device)

            if any(p == "" for p in prompt):
                empty_rows = torch.tensor([p == "" for p in prompt], dtype=torch.bool, device=text_encoder_device)
                input_ids[empty_rows] = bot_token_id
                attention_mask[empty_rows] = 1

        encoder_outputs = self.text_encoder(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = encoder_outputs.hidden_states

        prompt_embeds = torch.cat([hidden_states[-1], hidden_states[-2]], dim=-1)
        prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)

        prompt_embeds = prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)
        hidden_states = tuple(
            layer.repeat_interleave(num_images_per_prompt, dim=0).to(device=device) for layer in hidden_states
        )
        attention_mask = attention_mask.repeat_interleave(num_images_per_prompt, dim=0).to(device=device)

        return prompt_embeds, hidden_states, attention_mask

    @staticmethod
    def pad_embedding(prompt_embeds, max_tokens, attention_mask=None):
        # Pad embeddings to `max_tokens` while preserving the mask of real tokens.
        batch_size, seq_len, dim = prompt_embeds.shape

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_len), dtype=prompt_embeds.dtype, device=prompt_embeds.device)
        else:
            attention_mask = attention_mask.to(device=prompt_embeds.device, dtype=prompt_embeds.dtype)

        if max_tokens < seq_len:
            raise ValueError("`max_tokens` must be greater or equal to the current sequence length.")

        if max_tokens > seq_len:
            pad_length = max_tokens - seq_len
            padding = torch.zeros(
                (batch_size, pad_length, dim), dtype=prompt_embeds.dtype, device=prompt_embeds.device
            )
            prompt_embeds = torch.cat([prompt_embeds, padding], dim=1)

            mask_padding = torch.zeros(
                (batch_size, pad_length), dtype=prompt_embeds.dtype, device=prompt_embeds.device
            )
            attention_mask = torch.cat([attention_mask, mask_padding], dim=1)

        return prompt_embeds, attention_mask

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        guidance_scale: float = 5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        max_sequence_length: int = 3000,
        lora_scale: Optional[float] = None,
    ):
        r"""
        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            guidance_scale (`float`):
                Guidance scale for classifier free guidance.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
        """
        device = device or self._execution_device

        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, FluxLoraLoaderMixin):
            self._lora_scale = lora_scale

            # dynamically adjust the LoRA scale
            if self.text_encoder is not None and USE_PEFT_BACKEND:
                scale_lora_layers(self.text_encoder, lora_scale)

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        prompt_attention_mask = None
        negative_prompt_attention_mask = None
        if prompt_embeds is None:
            prompt_embeds, prompt_layers, prompt_attention_mask = self.get_prompt_embeds(
                prompt=prompt,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
            )
            prompt_embeds = prompt_embeds.to(dtype=self.transformer.dtype)
            prompt_layers = [tensor.to(dtype=self.transformer.dtype) for tensor in prompt_layers]

        if guidance_scale > 1:
            if isinstance(negative_prompt, list) and negative_prompt[0] is None:
                negative_prompt = ""
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_embeds, negative_prompt_layers, negative_prompt_attention_mask = self.get_prompt_embeds(
                prompt=negative_prompt,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
            )
            negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.transformer.dtype)
            negative_prompt_layers = [tensor.to(dtype=self.transformer.dtype) for tensor in negative_prompt_layers]

        if self.text_encoder is not None:
            if isinstance(self, FluxLoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(self.text_encoder, lora_scale)

        # Pad to longest
        if prompt_attention_mask is not None:
            prompt_attention_mask = prompt_attention_mask.to(device=prompt_embeds.device, dtype=prompt_embeds.dtype)

        if negative_prompt_embeds is not None:
            if negative_prompt_attention_mask is not None:
                negative_prompt_attention_mask = negative_prompt_attention_mask.to(
                    device=negative_prompt_embeds.device, dtype=negative_prompt_embeds.dtype
                )
            max_tokens = max(negative_prompt_embeds.shape[1], prompt_embeds.shape[1])

            prompt_embeds, prompt_attention_mask = self.pad_embedding(
                prompt_embeds, max_tokens, attention_mask=prompt_attention_mask
            )
            prompt_layers = [self.pad_embedding(layer, max_tokens)[0] for layer in prompt_layers]

            negative_prompt_embeds, negative_prompt_attention_mask = self.pad_embedding(
                negative_prompt_embeds, max_tokens, attention_mask=negative_prompt_attention_mask
            )
            negative_prompt_layers = [self.pad_embedding(layer, max_tokens)[0] for layer in negative_prompt_layers]
        else:
            max_tokens = prompt_embeds.shape[1]
            prompt_embeds, prompt_attention_mask = self.pad_embedding(
                prompt_embeds, max_tokens, attention_mask=prompt_attention_mask
            )
            negative_prompt_layers = None

        dtype = self.text_encoder.dtype
        text_ids = torch.zeros(prompt_embeds.shape[0], max_tokens, 3).to(device=device, dtype=dtype)

        return (
            prompt_embeds,
            negative_prompt_embeds,
            text_ids,
            prompt_attention_mask,
            negative_prompt_attention_mask,
            prompt_layers,
            negative_prompt_layers,
        )

    @property
    def guidance_scale(self):
        return self._guidance_scale

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.

    @property
    def joint_attention_kwargs(self):
        return self._joint_attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt

    @staticmethod
    # Based on diffusers.pipelines.flux.pipeline_flux.FluxPipeline._unpack_latents
    def _unpack_latents(latents, height, width, vae_scale_factor):
        batch_size, num_patches, channels = latents.shape

        height = height // vae_scale_factor
        width = width // vae_scale_factor

        latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)

        latents = latents.reshape(batch_size, channels // (2 * 2), height, width)
        return latents

    @staticmethod
    # Copied from diffusers.pipelines.flux.pipeline_flux.FluxPipeline._prepare_latent_image_ids
    def _prepare_latent_image_ids(batch_size, height, width, device, dtype):
        latent_image_ids = torch.zeros(height, width, 3)
        latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]

        latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

        latent_image_ids = latent_image_ids.reshape(
            latent_image_id_height * latent_image_id_width, latent_image_id_channels
        )

        return latent_image_ids.to(device=device, dtype=dtype)

    @staticmethod
    def _unpack_latents_no_patch(latents, height, width, vae_scale_factor):
        batch_size, num_patches, channels = latents.shape

        height = height // vae_scale_factor
        width = width // vae_scale_factor

        latents = latents.view(batch_size, height, width, channels)
        latents = latents.permute(0, 3, 1, 2)

        return latents

    @staticmethod
    def _pack_latents_no_patch(latents, batch_size, num_channels_latents, height, width):
        latents = latents.permute(0, 2, 3, 1)
        latents = latents.reshape(batch_size, height * width, num_channels_latents)
        return latents

    @staticmethod
    # Copied from diffusers.pipelines.flux.pipeline_flux.FluxPipeline._pack_latents
    def _pack_latents(latents, batch_size, num_channels_latents, height, width):
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

        return latents

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
        do_patching=False,
    ):
        height = int(height) // self.vae_scale_factor
        width = int(width) // self.vae_scale_factor

        shape = (batch_size, num_channels_latents, height, width)

        if latents is not None:
            latent_image_ids = self._prepare_latent_image_ids(batch_size, height, width, device, dtype)
            return latents.to(device=device, dtype=dtype), latent_image_ids

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        if do_patching:
            latents = self._pack_latents(latents, batch_size, num_channels_latents, height, width)
            latent_image_ids = self._prepare_latent_image_ids(batch_size, height // 2, width // 2, device, dtype)
        else:
            latents = self._pack_latents_no_patch(latents, batch_size, num_channels_latents, height, width)
            latent_image_ids = self._prepare_latent_image_ids(batch_size, height, width, device, dtype)

        return latents, latent_image_ids

    @staticmethod
    def _prepare_attention_mask(attention_mask):
        attention_matrix = torch.einsum("bi,bj->bij", attention_mask, attention_mask)

        # convert to 0 - keep, -inf ignore
        attention_matrix = torch.where(
            attention_matrix == 1, 0.0, -torch.inf
        )  # Apply -inf to ignored tokens for nulling softmax score
        return attention_matrix

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 30,
        timesteps: List[int] = None,
        guidance_scale: float = 5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 3000,
        do_patching=False,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image. This is set to 1024 by default for the best results.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image. This is set to 1024 by default for the best results.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            guidance_scale (`float`, *optional*, defaults to 5.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] instead
                of a plain tuple.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int` defaults to 3000): Maximum sequence length to use with the `prompt`.
            do_patching (`bool`, *optional*, defaults to `False`): Whether to use patching.
        Examples:
          Returns:
            [`~pipelines.flux.BriaFiboPipelineOutput`] or `tuple`: [`~pipelines.flux.BriaFiboPipelineOutput`] if
            `return_dict` is True, otherwise a `tuple`. When returning a tuple, the first element is a list with the
            generated images.
        """

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt=prompt,
            height=height,
            width=width,
            prompt_embeds=prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        lora_scale = (
            self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        )

        (
            prompt_embeds,
            negative_prompt_embeds,
            text_ids,
            prompt_attention_mask,
            negative_prompt_attention_mask,
            prompt_layers,
            negative_prompt_layers,
        ) = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            device=device,
            max_sequence_length=max_sequence_length,
            num_images_per_prompt=num_images_per_prompt,
            lora_scale=lora_scale,
        )
        prompt_batch_size = prompt_embeds.shape[0]

        if guidance_scale > 1:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            prompt_layers = [
                torch.cat([negative_prompt_layers[i], prompt_layers[i]], dim=0) for i in range(len(prompt_layers))
            ]
            prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)

        total_num_layers_transformer = len(self.transformer.transformer_blocks) + len(
            self.transformer.single_transformer_blocks
        )
        if len(prompt_layers) >= total_num_layers_transformer:
            # remove first layers
            prompt_layers = prompt_layers[len(prompt_layers) - total_num_layers_transformer :]
        else:
            # duplicate last layer
            prompt_layers = prompt_layers + [prompt_layers[-1]] * (total_num_layers_transformer - len(prompt_layers))

        # 5. Prepare latent variables

        num_channels_latents = self.transformer.config.in_channels
        if do_patching:
            num_channels_latents = int(num_channels_latents / 4)

        latents, latent_image_ids = self.prepare_latents(
            prompt_batch_size,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
            do_patching,
        )

        latent_attention_mask = torch.ones(
            [latents.shape[0], latents.shape[1]], dtype=latents.dtype, device=latents.device
        )
        if guidance_scale > 1:
            latent_attention_mask = latent_attention_mask.repeat(2, 1)

        attention_mask = torch.cat([prompt_attention_mask, latent_attention_mask], dim=1)
        attention_mask = self._prepare_attention_mask(attention_mask)  # batch, seq => batch, seq, seq
        attention_mask = attention_mask.unsqueeze(dim=1).to(dtype=self.transformer.dtype)  # for head broadcasting

        if self._joint_attention_kwargs is None:
            self._joint_attention_kwargs = {}
        self._joint_attention_kwargs["attention_mask"] = attention_mask

        # Adapt scheduler to dynamic shifting (resolution dependent)

        if do_patching:
            seq_len = (height // (self.vae_scale_factor * 2)) * (width // (self.vae_scale_factor * 2))
        else:
            seq_len = (height // self.vae_scale_factor) * (width // self.vae_scale_factor)

        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)

        mu = calculate_shift(
            seq_len,
            self.scheduler.config.base_image_seq_len,
            self.scheduler.config.max_image_seq_len,
            self.scheduler.config.base_shift,
            self.scheduler.config.max_shift,
        )

        # Init sigmas and timesteps according to shift size
        # This changes the scheduler in-place according to the dynamic scheduling
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps=num_inference_steps,
            device=device,
            timesteps=None,
            sigmas=sigmas,
            mu=mu,
        )

        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # Support old different diffusers versions
        if len(latent_image_ids.shape) == 3:
            latent_image_ids = latent_image_ids[0]

        if len(text_ids.shape) == 3:
            text_ids = text_ids[0]

        # 6. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1 else latents

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0]).to(
                    device=latent_model_input.device, dtype=latent_model_input.dtype
                )

                # This is predicts "v" from flow-matching or eps from diffusion
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    text_encoder_layers=prompt_layers,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                )[0]

                # perform guidance
                if guidance_scale > 1:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        if output_type == "latent":
            image = latents

        else:
            if do_patching:
                latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            else:
                latents = self._unpack_latents_no_patch(latents, height, width, self.vae_scale_factor)

            latents = latents.unsqueeze(dim=2)
            latents_device = latents[0].device
            latents_dtype = latents[0].dtype
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents_device, latents_dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                latents_device, latents_dtype
            )
            latents_scaled = [latent / latents_std + latents_mean for latent in latents]
            latents_scaled = torch.cat(latents_scaled, dim=0)
            image = []
            for scaled_latent in latents_scaled:
                curr_image = self.vae.decode(scaled_latent.unsqueeze(0), return_dict=False)[0]
                curr_image = self.image_processor.postprocess(curr_image.squeeze(dim=2), output_type=output_type)
                image.append(curr_image)
            if len(image) == 1:
                image = image[0]
            else:
                image = np.stack(image, axis=0)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return BriaFiboPipelineOutput(images=image)

    def check_inputs(
        self,
        prompt,
        height,
        width,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
        max_sequence_length=None,
    ):
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 16 but are {height} and {width}.")

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

        if max_sequence_length is not None and max_sequence_length > 3000:
            raise ValueError(f"`max_sequence_length` cannot be greater than 3000 but is {max_sequence_length}")
