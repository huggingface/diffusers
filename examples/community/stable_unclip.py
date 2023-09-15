import types
from typing import List, Optional, Tuple, Union

import torch
from transformers import CLIPTextModelWithProjection, CLIPTokenizer
from transformers.models.clip.modeling_clip import CLIPTextModelOutput

from diffusers.models import PriorTransformer
from diffusers.pipelines import DiffusionPipeline, StableDiffusionImageVariationPipeline
from diffusers.schedulers import UnCLIPScheduler
from diffusers.utils import logging
from diffusers.utils.torch_utils import randn_tensor


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def _encode_image(self, image, device, num_images_per_prompt, do_classifier_free_guidance):
    image = image.to(device=device)
    image_embeddings = image  # take image as image_embeddings
    image_embeddings = image_embeddings.unsqueeze(1)

    # duplicate image embeddings for each generation per prompt, using mps friendly method
    bs_embed, seq_len, _ = image_embeddings.shape
    image_embeddings = image_embeddings.repeat(1, num_images_per_prompt, 1)
    image_embeddings = image_embeddings.view(bs_embed * num_images_per_prompt, seq_len, -1)

    if do_classifier_free_guidance:
        uncond_embeddings = torch.zeros_like(image_embeddings)

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        image_embeddings = torch.cat([uncond_embeddings, image_embeddings])

    return image_embeddings


class StableUnCLIPPipeline(DiffusionPipeline):
    def __init__(
        self,
        prior: PriorTransformer,
        tokenizer: CLIPTokenizer,
        text_encoder: CLIPTextModelWithProjection,
        prior_scheduler: UnCLIPScheduler,
        decoder_pipe_kwargs: Optional[dict] = None,
    ):
        super().__init__()

        decoder_pipe_kwargs = {"image_encoder": None} if decoder_pipe_kwargs is None else decoder_pipe_kwargs

        decoder_pipe_kwargs["torch_dtype"] = decoder_pipe_kwargs.get("torch_dtype", None) or prior.dtype

        self.decoder_pipe = StableDiffusionImageVariationPipeline.from_pretrained(
            "lambdalabs/sd-image-variations-diffusers", **decoder_pipe_kwargs
        )

        # replace `_encode_image` method
        self.decoder_pipe._encode_image = types.MethodType(_encode_image, self.decoder_pipe)

        self.register_modules(
            prior=prior,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            prior_scheduler=prior_scheduler,
        )

    def _encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        text_model_output: Optional[Union[CLIPTextModelOutput, Tuple]] = None,
        text_attention_mask: Optional[torch.Tensor] = None,
    ):
        if text_model_output is None:
            batch_size = len(prompt) if isinstance(prompt, list) else 1
            # get prompt text embeddings
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            text_mask = text_inputs.attention_mask.bool().to(device)

            if text_input_ids.shape[-1] > self.tokenizer.model_max_length:
                removed_text = self.tokenizer.batch_decode(text_input_ids[:, self.tokenizer.model_max_length :])
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )
                text_input_ids = text_input_ids[:, : self.tokenizer.model_max_length]

            text_encoder_output = self.text_encoder(text_input_ids.to(device))

            text_embeddings = text_encoder_output.text_embeds
            text_encoder_hidden_states = text_encoder_output.last_hidden_state

        else:
            batch_size = text_model_output[0].shape[0]
            text_embeddings, text_encoder_hidden_states = text_model_output[0], text_model_output[1]
            text_mask = text_attention_mask

        text_embeddings = text_embeddings.repeat_interleave(num_images_per_prompt, dim=0)
        text_encoder_hidden_states = text_encoder_hidden_states.repeat_interleave(num_images_per_prompt, dim=0)
        text_mask = text_mask.repeat_interleave(num_images_per_prompt, dim=0)

        if do_classifier_free_guidance:
            uncond_tokens = [""] * batch_size

            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            uncond_text_mask = uncond_input.attention_mask.bool().to(device)
            uncond_embeddings_text_encoder_output = self.text_encoder(uncond_input.input_ids.to(device))

            uncond_embeddings = uncond_embeddings_text_encoder_output.text_embeds
            uncond_text_encoder_hidden_states = uncond_embeddings_text_encoder_output.last_hidden_state

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method

            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.repeat(1, num_images_per_prompt)
            uncond_embeddings = uncond_embeddings.view(batch_size * num_images_per_prompt, seq_len)

            seq_len = uncond_text_encoder_hidden_states.shape[1]
            uncond_text_encoder_hidden_states = uncond_text_encoder_hidden_states.repeat(1, num_images_per_prompt, 1)
            uncond_text_encoder_hidden_states = uncond_text_encoder_hidden_states.view(
                batch_size * num_images_per_prompt, seq_len, -1
            )
            uncond_text_mask = uncond_text_mask.repeat_interleave(num_images_per_prompt, dim=0)

            # done duplicates

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
            text_encoder_hidden_states = torch.cat([uncond_text_encoder_hidden_states, text_encoder_hidden_states])

            text_mask = torch.cat([uncond_text_mask, text_mask])

        return text_embeddings, text_encoder_hidden_states, text_mask

    @property
    def _execution_device(self):
        r"""
        Returns the device on which the pipeline's models will be executed. After calling
        `pipeline.enable_sequential_cpu_offload()` the execution device can only be inferred from Accelerate's module
        hooks.
        """
        if self.device != torch.device("meta") or not hasattr(self.prior, "_hf_hook"):
            return self.device
        for module in self.prior.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def prepare_latents(self, shape, dtype, device, generator, latents, scheduler):
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents.to(device)

        latents = latents * scheduler.init_noise_sigma
        return latents

    def to(self, torch_device: Optional[Union[str, torch.device]] = None):
        self.decoder_pipe.to(torch_device)
        super().to(torch_device)

    @torch.no_grad()
    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_images_per_prompt: int = 1,
        prior_num_inference_steps: int = 25,
        generator: Optional[torch.Generator] = None,
        prior_latents: Optional[torch.FloatTensor] = None,
        text_model_output: Optional[Union[CLIPTextModelOutput, Tuple]] = None,
        text_attention_mask: Optional[torch.Tensor] = None,
        prior_guidance_scale: float = 4.0,
        decoder_guidance_scale: float = 8.0,
        decoder_num_inference_steps: int = 50,
        decoder_num_images_per_prompt: Optional[int] = 1,
        decoder_eta: float = 0.0,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
    ):
        if prompt is not None:
            if isinstance(prompt, str):
                batch_size = 1
            elif isinstance(prompt, list):
                batch_size = len(prompt)
            else:
                raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        else:
            batch_size = text_model_output[0].shape[0]

        device = self._execution_device

        batch_size = batch_size * num_images_per_prompt

        do_classifier_free_guidance = prior_guidance_scale > 1.0 or decoder_guidance_scale > 1.0

        text_embeddings, text_encoder_hidden_states, text_mask = self._encode_prompt(
            prompt, device, num_images_per_prompt, do_classifier_free_guidance, text_model_output, text_attention_mask
        )

        # prior

        self.prior_scheduler.set_timesteps(prior_num_inference_steps, device=device)
        prior_timesteps_tensor = self.prior_scheduler.timesteps

        embedding_dim = self.prior.config.embedding_dim

        prior_latents = self.prepare_latents(
            (batch_size, embedding_dim),
            text_embeddings.dtype,
            device,
            generator,
            prior_latents,
            self.prior_scheduler,
        )

        for i, t in enumerate(self.progress_bar(prior_timesteps_tensor)):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([prior_latents] * 2) if do_classifier_free_guidance else prior_latents

            predicted_image_embedding = self.prior(
                latent_model_input,
                timestep=t,
                proj_embedding=text_embeddings,
                encoder_hidden_states=text_encoder_hidden_states,
                attention_mask=text_mask,
            ).predicted_image_embedding

            if do_classifier_free_guidance:
                predicted_image_embedding_uncond, predicted_image_embedding_text = predicted_image_embedding.chunk(2)
                predicted_image_embedding = predicted_image_embedding_uncond + prior_guidance_scale * (
                    predicted_image_embedding_text - predicted_image_embedding_uncond
                )

            if i + 1 == prior_timesteps_tensor.shape[0]:
                prev_timestep = None
            else:
                prev_timestep = prior_timesteps_tensor[i + 1]

            prior_latents = self.prior_scheduler.step(
                predicted_image_embedding,
                timestep=t,
                sample=prior_latents,
                generator=generator,
                prev_timestep=prev_timestep,
            ).prev_sample

        prior_latents = self.prior.post_process_latents(prior_latents)

        image_embeddings = prior_latents

        output = self.decoder_pipe(
            image=image_embeddings,
            height=height,
            width=width,
            num_inference_steps=decoder_num_inference_steps,
            guidance_scale=decoder_guidance_scale,
            generator=generator,
            output_type=output_type,
            return_dict=return_dict,
            num_images_per_prompt=decoder_num_images_per_prompt,
            eta=decoder_eta,
        )
        return output
