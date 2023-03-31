import inspect
from typing import List, Optional, Union

import PIL
import torch
from torch.nn import functional as F
from transformers import (
    CLIPImageProcessor,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)

from diffusers import (
    DiffusionPipeline,
    ImagePipelineOutput,
    UnCLIPScheduler,
    UNet2DConditionModel,
    UNet2DModel,
)
from diffusers.pipelines.unclip import UnCLIPTextProjModel
from diffusers.utils import is_accelerate_available, logging, randn_tensor


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def slerp(val, low, high):
    """
    Find the interpolation point between the 'low' and 'high' values for the given 'val'. See https://en.wikipedia.org/wiki/Slerp for more details on the topic.
    """
    low_norm = low / torch.norm(low)
    high_norm = high / torch.norm(high)
    omega = torch.acos((low_norm * high_norm))
    so = torch.sin(omega)
    res = (torch.sin((1.0 - val) * omega) / so) * low + (torch.sin(val * omega) / so) * high
    return res


class UnCLIPImageInterpolationPipeline(DiffusionPipeline):
    """
    Pipeline to generate variations from an input image using unCLIP

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        text_encoder ([`CLIPTextModelWithProjection`]):
            Frozen text-encoder.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        feature_extractor ([`CLIPImageProcessor`]):
            Model that extracts features from generated images to be used as inputs for the `image_encoder`.
        image_encoder ([`CLIPVisionModelWithProjection`]):
            Frozen CLIP image-encoder. unCLIP Image Variation uses the vision portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPVisionModelWithProjection),
            specifically the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        text_proj ([`UnCLIPTextProjModel`]):
            Utility class to prepare and combine the embeddings before they are passed to the decoder.
        decoder ([`UNet2DConditionModel`]):
            The decoder to invert the image embedding into an image.
        super_res_first ([`UNet2DModel`]):
            Super resolution unet. Used in all but the last step of the super resolution diffusion process.
        super_res_last ([`UNet2DModel`]):
            Super resolution unet. Used in the last step of the super resolution diffusion process.
        decoder_scheduler ([`UnCLIPScheduler`]):
            Scheduler used in the decoder denoising process. Just a modified DDPMScheduler.
        super_res_scheduler ([`UnCLIPScheduler`]):
            Scheduler used in the super resolution denoising process. Just a modified DDPMScheduler.

    """

    decoder: UNet2DConditionModel
    text_proj: UnCLIPTextProjModel
    text_encoder: CLIPTextModelWithProjection
    tokenizer: CLIPTokenizer
    feature_extractor: CLIPImageProcessor
    image_encoder: CLIPVisionModelWithProjection
    super_res_first: UNet2DModel
    super_res_last: UNet2DModel

    decoder_scheduler: UnCLIPScheduler
    super_res_scheduler: UnCLIPScheduler

    # Copied from diffusers.pipelines.unclip.pipeline_unclip_image_variation.UnCLIPImageVariationPipeline.__init__
    def __init__(
        self,
        decoder: UNet2DConditionModel,
        text_encoder: CLIPTextModelWithProjection,
        tokenizer: CLIPTokenizer,
        text_proj: UnCLIPTextProjModel,
        feature_extractor: CLIPImageProcessor,
        image_encoder: CLIPVisionModelWithProjection,
        super_res_first: UNet2DModel,
        super_res_last: UNet2DModel,
        decoder_scheduler: UnCLIPScheduler,
        super_res_scheduler: UnCLIPScheduler,
    ):
        super().__init__()

        self.register_modules(
            decoder=decoder,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_proj=text_proj,
            feature_extractor=feature_extractor,
            image_encoder=image_encoder,
            super_res_first=super_res_first,
            super_res_last=super_res_last,
            decoder_scheduler=decoder_scheduler,
            super_res_scheduler=super_res_scheduler,
        )

    # Copied from diffusers.pipelines.unclip.pipeline_unclip.UnCLIPPipeline.prepare_latents
    def prepare_latents(self, shape, dtype, device, generator, latents, scheduler):
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents.to(device)

        latents = latents * scheduler.init_noise_sigma
        return latents

    # Copied from diffusers.pipelines.unclip.pipeline_unclip_image_variation.UnCLIPImageVariationPipeline._encode_prompt
    def _encode_prompt(self, prompt, device, num_images_per_prompt, do_classifier_free_guidance):
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
        text_encoder_output = self.text_encoder(text_input_ids.to(device))

        prompt_embeds = text_encoder_output.text_embeds
        text_encoder_hidden_states = text_encoder_output.last_hidden_state

        prompt_embeds = prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)
        text_encoder_hidden_states = text_encoder_hidden_states.repeat_interleave(num_images_per_prompt, dim=0)
        text_mask = text_mask.repeat_interleave(num_images_per_prompt, dim=0)

        if do_classifier_free_guidance:
            uncond_tokens = [""] * batch_size

            max_length = text_input_ids.shape[-1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )
            uncond_text_mask = uncond_input.attention_mask.bool().to(device)
            negative_prompt_embeds_text_encoder_output = self.text_encoder(uncond_input.input_ids.to(device))

            negative_prompt_embeds = negative_prompt_embeds_text_encoder_output.text_embeds
            uncond_text_encoder_hidden_states = negative_prompt_embeds_text_encoder_output.last_hidden_state

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method

            seq_len = negative_prompt_embeds.shape[1]
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len)

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
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            text_encoder_hidden_states = torch.cat([uncond_text_encoder_hidden_states, text_encoder_hidden_states])

            text_mask = torch.cat([uncond_text_mask, text_mask])

        return prompt_embeds, text_encoder_hidden_states, text_mask

    # Copied from diffusers.pipelines.unclip.pipeline_unclip_image_variation.UnCLIPImageVariationPipeline._encode_image
    def _encode_image(self, image, device, num_images_per_prompt, image_embeddings: Optional[torch.Tensor] = None):
        dtype = next(self.image_encoder.parameters()).dtype

        if image_embeddings is None:
            if not isinstance(image, torch.Tensor):
                image = self.feature_extractor(images=image, return_tensors="pt").pixel_values

            image = image.to(device=device, dtype=dtype)
            image_embeddings = self.image_encoder(image).image_embeds

        image_embeddings = image_embeddings.repeat_interleave(num_images_per_prompt, dim=0)

        return image_embeddings

    # Copied from diffusers.pipelines.unclip.pipeline_unclip_image_variation.UnCLIPImageVariationPipeline.enable_sequential_cpu_offload
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
            self.decoder,
            self.text_proj,
            self.text_encoder,
            self.super_res_first,
            self.super_res_last,
        ]
        for cpu_offloaded_model in models:
            if cpu_offloaded_model is not None:
                cpu_offload(cpu_offloaded_model, device)

    @property
    # Copied from diffusers.pipelines.unclip.pipeline_unclip.UnCLIPPipeline._execution_device
    def _execution_device(self):
        r"""
        Returns the device on which the pipeline's models will be executed. After calling
        `pipeline.enable_sequential_cpu_offload()` the execution device can only be inferred from Accelerate's module
        hooks.
        """
        if self.device != torch.device("meta") or not hasattr(self.decoder, "_hf_hook"):
            return self.device
        for module in self.decoder.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    @torch.no_grad()
    def __call__(
        self,
        image: Optional[Union[List[PIL.Image.Image], torch.FloatTensor]] = None,
        steps: int = 5,
        decoder_num_inference_steps: int = 25,
        super_res_num_inference_steps: int = 7,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        image_embeddings: Optional[torch.Tensor] = None,
        decoder_latents: Optional[torch.FloatTensor] = None,
        super_res_latents: Optional[torch.FloatTensor] = None,
        decoder_guidance_scale: float = 8.0,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
    ):
        """
        Function invoked when calling the pipeline for generation.

        Args:
            image (`List[PIL.Image.Image]` or `torch.FloatTensor`):
                The images to use for the image interpolation. Only accepts a list of two PIL Images or If you provide a tensor, it needs to comply with the
                configuration of
                [this](https://huggingface.co/fusing/karlo-image-variations-diffusers/blob/main/feature_extractor/preprocessor_config.json)
                `CLIPImageProcessor` while still having a shape of two in the 0th dimension. Can be left to `None` only when `image_embeddings` are passed.
            steps (`int`, *optional*, defaults to 5):
                The number of interpolation images to generate.
            decoder_num_inference_steps (`int`, *optional*, defaults to 25):
                The number of denoising steps for the decoder. More denoising steps usually lead to a higher quality
                image at the expense of slower inference.
            super_res_num_inference_steps (`int`, *optional*, defaults to 7):
                The number of denoising steps for super resolution. More denoising steps usually lead to a higher
                quality image at the expense of slower inference.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            image_embeddings (`torch.Tensor`, *optional*):
                Pre-defined image embeddings that can be derived from the image encoder. Pre-defined image embeddings
                can be passed for tasks like image interpolations. `image` can the be left to `None`.
            decoder_latents (`torch.FloatTensor` of shape (batch size, channels, height, width), *optional*):
                Pre-generated noisy latents to be used as inputs for the decoder.
            super_res_latents (`torch.FloatTensor` of shape (batch size, channels, super res height, super res width), *optional*):
                Pre-generated noisy latents to be used as inputs for the decoder.
            decoder_guidance_scale (`float`, *optional*, defaults to 4.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.
        """

        batch_size = steps

        device = self._execution_device

        if isinstance(image, List):
            if len(image) != 2:
                raise AssertionError(
                    f"Expected 'image' List to be of size 2, but passed 'image' length is {len(image)}"
                )
            elif not (isinstance(image[0], PIL.Image.Image) and isinstance(image[0], PIL.Image.Image)):
                raise AssertionError(
                    f"Expected 'image' List to contain PIL.Image.Image, but passed 'image' contents are {type(image[0])} and {type(image[1])}"
                )
        elif isinstance(image, torch.FloatTensor):
            if image.shape[0] != 2:
                raise AssertionError(
                    f"Expected 'image' to be torch.FloatTensor of shape 2 in 0th dimension, but passed 'image' size is {image.shape[0]}"
                )
        elif isinstance(image_embeddings, torch.Tensor):
            if image_embeddings.shape[0] != 2:
                raise AssertionError(
                    f"Expected 'image_embeddings' to be torch.FloatTensor of shape 2 in 0th dimension, but passed 'image_embeddings' shape is {image_embeddings.shape[0]}"
                )
        else:
            raise AssertionError(
                f"Expected 'image' or 'image_embeddings' to be not None with types List[PIL.Image] or Torch.FloatTensor respectively. Received {type(image)} and {type(image_embeddings)} repsectively"
            )

        original_image_embeddings = self._encode_image(
            image=image, device=device, num_images_per_prompt=1, image_embeddings=image_embeddings
        )

        image_embeddings = []

        for interp_step in torch.linspace(0, 1, steps):
            temp_image_embeddings = slerp(
                interp_step, original_image_embeddings[0], original_image_embeddings[1]
            ).unsqueeze(0)
            image_embeddings.append(temp_image_embeddings)

        image_embeddings = torch.cat(image_embeddings).to(device)

        do_classifier_free_guidance = decoder_guidance_scale > 1.0

        prompt_embeds, text_encoder_hidden_states, text_mask = self._encode_prompt(
            prompt=["" for i in range(steps)],
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=do_classifier_free_guidance,
        )

        text_encoder_hidden_states, additive_clip_time_embeddings = self.text_proj(
            image_embeddings=image_embeddings,
            prompt_embeds=prompt_embeds,
            text_encoder_hidden_states=text_encoder_hidden_states,
            do_classifier_free_guidance=do_classifier_free_guidance,
        )

        if device.type == "mps":
            # HACK: MPS: There is a panic when padding bool tensors,
            # so cast to int tensor for the pad and back to bool afterwards
            text_mask = text_mask.type(torch.int)
            decoder_text_mask = F.pad(text_mask, (self.text_proj.clip_extra_context_tokens, 0), value=1)
            decoder_text_mask = decoder_text_mask.type(torch.bool)
        else:
            decoder_text_mask = F.pad(text_mask, (self.text_proj.clip_extra_context_tokens, 0), value=True)

        self.decoder_scheduler.set_timesteps(decoder_num_inference_steps, device=device)
        decoder_timesteps_tensor = self.decoder_scheduler.timesteps

        num_channels_latents = self.decoder.in_channels
        height = self.decoder.sample_size
        width = self.decoder.sample_size

        decoder_latents = self.prepare_latents(
            (batch_size, num_channels_latents, height, width),
            text_encoder_hidden_states.dtype,
            device,
            generator,
            decoder_latents,
            self.decoder_scheduler,
        )

        for i, t in enumerate(self.progress_bar(decoder_timesteps_tensor)):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([decoder_latents] * 2) if do_classifier_free_guidance else decoder_latents

            noise_pred = self.decoder(
                sample=latent_model_input,
                timestep=t,
                encoder_hidden_states=text_encoder_hidden_states,
                class_labels=additive_clip_time_embeddings,
                attention_mask=decoder_text_mask,
            ).sample

            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred_uncond, _ = noise_pred_uncond.split(latent_model_input.shape[1], dim=1)
                noise_pred_text, predicted_variance = noise_pred_text.split(latent_model_input.shape[1], dim=1)
                noise_pred = noise_pred_uncond + decoder_guidance_scale * (noise_pred_text - noise_pred_uncond)
                noise_pred = torch.cat([noise_pred, predicted_variance], dim=1)

            if i + 1 == decoder_timesteps_tensor.shape[0]:
                prev_timestep = None
            else:
                prev_timestep = decoder_timesteps_tensor[i + 1]

            # compute the previous noisy sample x_t -> x_t-1
            decoder_latents = self.decoder_scheduler.step(
                noise_pred, t, decoder_latents, prev_timestep=prev_timestep, generator=generator
            ).prev_sample

        decoder_latents = decoder_latents.clamp(-1, 1)

        image_small = decoder_latents

        # done decoder

        # super res

        self.super_res_scheduler.set_timesteps(super_res_num_inference_steps, device=device)
        super_res_timesteps_tensor = self.super_res_scheduler.timesteps

        channels = self.super_res_first.in_channels // 2
        height = self.super_res_first.sample_size
        width = self.super_res_first.sample_size

        super_res_latents = self.prepare_latents(
            (batch_size, channels, height, width),
            image_small.dtype,
            device,
            generator,
            super_res_latents,
            self.super_res_scheduler,
        )

        if device.type == "mps":
            # MPS does not support many interpolations
            image_upscaled = F.interpolate(image_small, size=[height, width])
        else:
            interpolate_antialias = {}
            if "antialias" in inspect.signature(F.interpolate).parameters:
                interpolate_antialias["antialias"] = True

            image_upscaled = F.interpolate(
                image_small, size=[height, width], mode="bicubic", align_corners=False, **interpolate_antialias
            )

        for i, t in enumerate(self.progress_bar(super_res_timesteps_tensor)):
            # no classifier free guidance

            if i == super_res_timesteps_tensor.shape[0] - 1:
                unet = self.super_res_last
            else:
                unet = self.super_res_first

            latent_model_input = torch.cat([super_res_latents, image_upscaled], dim=1)

            noise_pred = unet(
                sample=latent_model_input,
                timestep=t,
            ).sample

            if i + 1 == super_res_timesteps_tensor.shape[0]:
                prev_timestep = None
            else:
                prev_timestep = super_res_timesteps_tensor[i + 1]

            # compute the previous noisy sample x_t -> x_t-1
            super_res_latents = self.super_res_scheduler.step(
                noise_pred, t, super_res_latents, prev_timestep=prev_timestep, generator=generator
            ).prev_sample

        image = super_res_latents
        # done super res

        # post processing

        image = image * 0.5 + 0.5
        image = image.clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()

        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)
