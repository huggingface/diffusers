from logging import getLogger
from typing import Any, Callable, List, Optional, Union

import numpy as np
import PIL
import torch

from ...schedulers import DDPMScheduler
from ..onnx_utils import ORT_TO_NP_TYPE, OnnxRuntimeModel
from ..pipeline_utils import ImagePipelineOutput
from . import StableDiffusionUpscalePipeline


logger = getLogger(__name__)


NUM_LATENT_CHANNELS = 4
NUM_UNET_INPUT_CHANNELS = 7

ORT_TO_PT_TYPE = {
    "float16": torch.float16,
    "float32": torch.float32,
}


def preprocess(image):
    if isinstance(image, torch.Tensor):
        return image
    elif isinstance(image, PIL.Image.Image):
        image = [image]

    if isinstance(image[0], PIL.Image.Image):
        w, h = image[0].size
        w, h = map(lambda x: x - x % 64, (w, h))  # resize to integer multiple of 32

        image = [np.array(i.resize((w, h)))[None, :] for i in image]
        image = np.concatenate(image, axis=0)
        image = np.array(image).astype(np.float32) / 255.0
        image = image.transpose(0, 3, 1, 2)
        image = 2.0 * image - 1.0
        image = torch.from_numpy(image)
    elif isinstance(image[0], torch.Tensor):
        image = torch.cat(image, dim=0)

    return image


class OnnxStableDiffusionUpscalePipeline(StableDiffusionUpscalePipeline):
    def __init__(
        self,
        vae: OnnxRuntimeModel,
        text_encoder: OnnxRuntimeModel,
        tokenizer: Any,
        unet: OnnxRuntimeModel,
        low_res_scheduler: DDPMScheduler,
        scheduler: Any,
        max_noise_level: int = 350,
    ):
        super().__init__(vae, text_encoder, tokenizer, unet, low_res_scheduler, scheduler, max_noise_level)

    def __call__(
        self,
        prompt: Union[str, List[str]],
        image: Union[torch.FloatTensor, PIL.Image.Image, List[PIL.Image.Image]],
        num_inference_steps: int = 75,
        guidance_scale: float = 9.0,
        noise_level: int = 20,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
    ):
        # 1. Check inputs
        self.check_inputs(prompt, image, noise_level, callback_steps)

        # 2. Define call parameters
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_embeddings = self._encode_prompt(
            prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
        )

        latents_dtype = ORT_TO_PT_TYPE[str(text_embeddings.dtype)]

        # 4. Preprocess image
        image = preprocess(image)
        image = image.cpu()

        # 5. set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Add noise to image
        noise_level = torch.tensor([noise_level], dtype=torch.long, device=device)
        noise = torch.randn(image.shape, generator=generator, device=device, dtype=latents_dtype)
        image = self.low_res_scheduler.add_noise(image, noise, noise_level)

        batch_multiplier = 2 if do_classifier_free_guidance else 1
        image = np.concatenate([image] * batch_multiplier * num_images_per_prompt)
        noise_level = np.concatenate([noise_level] * image.shape[0])

        # 6. Prepare latent variables
        height, width = image.shape[2:]
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            NUM_LATENT_CHANNELS,
            height,
            width,
            latents_dtype,
            device,
            generator,
            latents,
        )

        # 7. Check that sizes of image and latents match
        num_channels_image = image.shape[1]
        if NUM_LATENT_CHANNELS + num_channels_image != NUM_UNET_INPUT_CHANNELS:
            raise ValueError(
                "Incorrect configuration settings! The config of `pipeline.unet` expects"
                f" {NUM_UNET_INPUT_CHANNELS} but received `num_channels_latents`: {NUM_LATENT_CHANNELS} +"
                f" `num_channels_image`: {num_channels_image} "
                f" = {NUM_LATENT_CHANNELS+num_channels_image}. Please verify the config of"
                " `pipeline.unet` or your `image` input."
            )

        # 8. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        timestep_dtype = next(
            (input.type for input in self.unet.model.get_inputs() if input.name == "timestep"), "tensor(float)"
        )
        timestep_dtype = ORT_TO_NP_TYPE[timestep_dtype]

        # 9. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = np.concatenate([latents] * 2) if do_classifier_free_guidance else latents

                # concat latents, mask, masked_image_latents in the channel dimension
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                latent_model_input = np.concatenate([latent_model_input, image], axis=1)

                # timestep to tensor
                timestep = np.array([t], dtype=timestep_dtype)

                # predict the noise residual
                noise_pred = self.unet(
                    sample=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=text_embeddings,
                    class_labels=noise_level.astype(np.int64),
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = np.split(noise_pred, 2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(
                    torch.from_numpy(noise_pred), t, latents, **extra_step_kwargs
                ).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        # 10. Post-processing
        image = self.decode_latents(latents.float())

        # 11. Convert to PIL
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)

    def decode_latents(self, latents):
        latents = 1 / 0.08333 * latents
        image = self.vae(latent_sample=latents)[0]
        image = np.clip(image / 2 + 0.5, 0, 1)
        image = image.transpose((0, 2, 3, 1))
        return image

    def _encode_prompt(self, prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt):
        batch_size = len(prompt) if isinstance(prompt, list) else 1

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

        # if hasattr(text_inputs, "attention_mask"):
        #     attention_mask = text_inputs.attention_mask.to(device)
        # else:
        #     attention_mask = None

        # no positional arguments to text_encoder
        text_embeddings = self.text_encoder(
            input_ids=text_input_ids.int().to(device),
            # attention_mask=attention_mask,
        )
        text_embeddings = text_embeddings[0]

        bs_embed, seq_len, _ = text_embeddings.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        text_embeddings = text_embeddings.repeat(1, num_images_per_prompt)
        text_embeddings = text_embeddings.reshape(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
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

            max_length = text_input_ids.shape[-1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            # if hasattr(uncond_input, "attention_mask"):
            #     attention_mask = uncond_input.attention_mask.to(device)
            # else:
            #     attention_mask = None

            uncond_embeddings = self.text_encoder(
                input_ids=uncond_input.input_ids.int().to(device),
                # attention_mask=attention_mask,
            )
            uncond_embeddings = uncond_embeddings[0]

            seq_len = uncond_embeddings.shape[1]
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            uncond_embeddings = uncond_embeddings.repeat(1, num_images_per_prompt)
            uncond_embeddings = uncond_embeddings.reshape(batch_size * num_images_per_prompt, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = np.concatenate([uncond_embeddings, text_embeddings])

        return text_embeddings
