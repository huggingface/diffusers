from typing import Callable, List, Optional, Tuple, Union

import torch

from diffusers import SpatialTransformer, VQModel
from diffusers.schedulers.scheduling_vq_diffusion import VQDiffusionScheduler
from transformers import CLIPTextModel, CLIPTokenizer

from ...pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from ...utils import logging


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class VQDiffusionPipeline(DiffusionPipeline):
    r"""
    Pipeline for text-to-image generation using VQ Diffusion

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        vqvae ([`VQModel`]):
            Vector Quantized Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent
            representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. VQ Diffusion uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        transformer (`SpatialTransformer`):
            Conditional transformer to denoise the encoded image latents.
        scheduler ([`VQDiffusionScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
    """

    vqvae: VQModel
    text_encoder: CLIPTextModel
    tokenizer: CLIPTokenizer
    transformer: SpatialTransformer
    scheduler: VQDiffusionScheduler

    def __init__(
        self,
        vqvae: VQModel,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        transformer: SpatialTransformer,
        scheduler: VQDiffusionScheduler,
    ):
        super().__init__()

        self.register_modules(
            vqvae=vqvae,
            transformer=transformer,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler,
        )

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        num_inference_steps: int = 100,
        truncation_rate: float = 1.0,
        num_images_per_prompt: int = 1,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
    ) -> Union[ImagePipelineOutput, Tuple]:
        """
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            num_inference_steps (`int`, *optional*, defaults to 100):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            truncation_rate (`float`, *optional*, defaults to 1.0 (equivalent to no truncation)):
                Used to "truncate" the predicted classes for x_0 such that the cumulative probability for a pixel is at
                most `truncation_rate`. The lowest probabilities that would increase the cumulative probability above
                `truncation_rate` are set to zero.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator`, *optional*):
                A [torch generator](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation
                deterministic.
            latents (`torch.FloatTensor` of shape (batch), *optional*):
                Pre-generated noisy latents to be used as inputs for image generation. Must be valid embedding indices.
                Can be used to tweak the same generation with different prompts. If not provided, a latents tensor will
                be generated of completely masked latent pixels.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.

        Returns:
            [`~pipeline_utils.ImagePipelineOutput`] or `tuple`: [`~pipelines.utils.ImagePipelineOutput`] if
            `return_dict` is True, otherwise a `tuple. When returning a tuple, the first element is a list with the
            generated images.
        """
        if isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        batch_size = batch_size * num_images_per_prompt

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

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
        text_embeddings = self.text_encoder(text_input_ids.to(self.device))[0]

        # NOTE: This additional step of normalizing the text embeddings is from VQ-Diffusion.
        # While CLIP does normalize the pooled output of the text transformer when combining
        # the image and text embeddings, CLIP does not directly normalize the last hidden state.
        #
        # CLIP normalizing the pooled output.
        # https://github.com/huggingface/transformers/blob/d92e22d1f28324f513f3080e5c47c071a3916721/src/transformers/models/clip/modeling_clip.py#L1052-L1053
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)

        # duplicate text embeddings for each generation per prompt
        text_embeddings = text_embeddings.repeat_interleave(num_images_per_prompt, dim=0)

        # get the initial completely masked latents unless the user supplied it

        latents_shape = (batch_size, self.transformer.num_latent_pixels)
        if latents is None:
            mask_class = self.transformer.num_embed - 1
            latents = torch.full(latents_shape, mask_class).to(self.device)
        else:
            if latents.shape != latents_shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {latents_shape}")
            if (latents < 0).any() or (latents >= self.transformer.num_embed).any():
                raise ValueError(
                    "Unexpected latents value(s). All latents be valid embedding indices i.e. in the range 0,"
                    f" {self.transformer.num_embed - 1} (inclusive)."
                )
            latents = latents.to(self.device)

        # set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)

        timesteps_tensor = self.scheduler.timesteps.to(self.device)

        x_t = latents

        for i, t in enumerate(self.progress_bar(timesteps_tensor)):
            # predict the un-noised image
            log_p_x_0 = self.transformer(hidden_states=x_t, context=text_embeddings, timestep=t)

            log_p_x_0 = self.truncate(log_p_x_0, truncation_rate)

            # remove `log(0)`'s (`-inf`s)
            log_p_x_0 = log_p_x_0.clamp(-70)

            # compute the previous noisy sample x_t -> x_t-1
            x_t = self.scheduler.step(log_p_x_0, t, x_t, generator=generator).prev_sample

            # call the callback, if provided
            if callback is not None and i % callback_steps == 0:
                callback(i, t, x_t)

        embedding_channels = self.vqvae.quantize.e_dim
        embeddings_shape = (batch_size, self.transformer.height, self.transformer.width, embedding_channels)
        embeddings = self.vqvae.quantize.get_codebook_entry(x_t, shape=embeddings_shape)
        image = self.vqvae.decode(embeddings, force_not_quantize=True).sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()

        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)

    def truncate(self, log_p_x_0: torch.FloatTensor, truncation_rate: float) -> torch.FloatTensor:
        """
        Truncates log_p_x_0 such that for each column vector, the total cumulative probability is `truncation_rate` The
        lowest probabilities that would increase the cumulative probability above `truncation_rate` are set to zero.
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
