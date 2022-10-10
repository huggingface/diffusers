from dataclasses import dataclass
from typing import Callable, List, Optional, Union

import numpy as np
import torch

import PIL
from diffusers import VQDiffusionTransformer, VQModel
from diffusers.schedulers.scheduling_vq_diffusion import VQDiffusionScheduler
from transformers import CLIPTextModel, CLIPTokenizer

from ...pipeline_utils import DiffusionPipeline
from ...utils import BaseOutput, logging


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class VQDiffusionPipelineOutput(BaseOutput):
    """
    Args:
    Output class for VQ Diffusion pipelines.
        images (`List[PIL.Image.Image]` or `np.ndarray`)
            List of denoised PIL images of length `batch_size` or numpy array of shape `(batch_size, height, width,
            num_channels)`. PIL images or numpy array present the denoised images of the diffusion pipeline.
    """

    images: Union[List[PIL.Image.Image], np.ndarray]


# This class is a placeholder and does not have the full VQ-diffusion pipeline built out yet
#
# NOTE: In VQ-Diffusion, the VQVAE trained on the ITHQ dataset uses an EMA variant of the vector quantizer
# in diffusers. The EMA variant uses EMA's to update the codebook during training but acts the same as the
# usual vector quantizer during inference. The VQDiffusion pipeline uses the non-ema vector quantizer during
# inference. If diffusers is to support training, the EMA vector quantizer could be implemented. For more
# information on EMA Vector quantizers, see https://arxiv.org/abs/1711.00937.
class VQDiffusionPipeline(DiffusionPipeline):
    vqvae: VQModel
    transformer: VQDiffusionTransformer

    def __init__(
        self,
        vqvae: VQModel,
        transformer: VQDiffusionTransformer,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
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
        height: int = 256,
        width: int = 256,
        num_inference_steps: int = 100,
        num_images_per_prompt: int = 1,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
    ):
        if isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

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

        # get the initial random noise unless the user supplied it

        # TODO HERE - what's the input shape?
        latents_shape = TODO  # (batch_size * num_images_per_prompt, self.unet.in_channels, height // 8, width // 8)
        latents_dtype = text_embeddings.dtype
        if latents is None:
            # all masked?
            latents = TODO  # torch.randn(latents_shape, generator=generator, device="cpu", dtype=latents_dtype).to(self.device)
        else:
            if latents.shape != latents_shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {latents_shape}")
            latents = latents.to(self.device)

        # set timesteps
        self.scheduler.set_timesteps(num_inference_steps)

        timesteps_tensor = self.scheduler.timesteps.to(self.device)

        for i, t in enumerate(self.progress_bar(timesteps_tensor)):
            # predict the un-noised image
            x0_pred = TODO  # self.transformer(latents, t, encoder_hidden_states=text_embeddings).sample

            # compute the previous noisy sample x_t -> x_t-1
            latents = TODO  # self.scheduler.step(x0_pred, t, latents).prev_sample

            # call the callback, if provided
            if callback is not None and i % callback_steps == 0:
                callback(i, t, latents)

        image = self.vqvae.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()

        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return image

        return VQDiffusionPipelineOutput(images=image)
