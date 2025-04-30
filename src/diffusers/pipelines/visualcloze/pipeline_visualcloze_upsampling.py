# Copyright 2025 VisualCloze team and The HuggingFace Team. All rights reserved.
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

import inspect
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

from .pipeline_visualcloze_generation import VisualClozeGenerationPipeline, calculate_shift, retrieve_latents, retrieve_timesteps
from ...loaders import FluxLoraLoaderMixin, FromSingleFileMixin, TextualInversionLoaderMixin
from ...models.autoencoders import AutoencoderKL
from ...models.transformers import FluxTransformer2DModel
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import (
    USE_PEFT_BACKEND,
    is_torch_xla_available,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from ...utils.torch_utils import randn_tensor
from ..flux.pipeline_output import FluxPipelineOutput
from ..pipeline_utils import DiffusionPipeline
from .visualcloze_utils import VisualClozeProcessor


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class VisualClozeUpsamplingPipeline(
    VisualClozeGenerationPipeline,
):
    r"""
    The VisualCloze pipeline for image generation with visual context. Reference:
    https://github.com/lzyhha/VisualCloze/tree/main This pipeline is designed to generate images based on visual
    in-context examples.

    Args:
        transformer ([`FluxTransformer2DModel`]):
            Conditional Transformer (MMDiT) architecture to denoise the encoded image latents.
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        text_encoder_2 ([`T5EncoderModel`]):
            [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5EncoderModel), specifically
            the [google/t5-v1_1-xxl](https://huggingface.co/google/t5-v1_1-xxl) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/en/model_doc/clip#transformers.CLIPTokenizer).
        tokenizer_2 (`T5TokenizerFast`):
            Second Tokenizer of class
            [T5TokenizerFast](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5TokenizerFast).
        resolution (`int`, *optional*, defaults to 384):
            The resolution of each image when concatenating images from the query and in-context examples.
    """

    model_cpu_offload_seq = "text_encoder->text_encoder_2->transformer->vae"
    _optional_components = []
    _callback_tensor_inputs = ["latents", "prompt_embeds"]


    def check_inputs(
        self,
        image,
        task_prompt,
        content_prompt,
        upsampling_height,
        upsampling_width,
        strength,
        prompt_embeds=None,
        pooled_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
        max_sequence_length=None,
    ):
        if strength < 0 or strength > 1:
            raise ValueError(f"The value of strength should in [0.0, 1.0] but is {strength}")

        if upsampling_height is not None and upsampling_height % (self.vae_scale_factor * 2) != 0:
            logger.warning(
                f"`upsampling_height`has to be divisible by {self.vae_scale_factor * 2} but are {upsampling_height}. Dimensions will be resized accordingly"
            )
        if upsampling_width is not None and upsampling_width % (self.vae_scale_factor * 2) != 0:
            logger.warning(
                f"`upsampling_width` have to be divisible by {self.vae_scale_factor * 2} but are {upsampling_width}. Dimensions will be resized accordingly"
            )

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        # Validate prompt inputs
        if (task_prompt is not None or content_prompt is not None) and prompt_embeds is not None:
            raise ValueError("Cannot provide both text `task_prompt` + `content_prompt` and `prompt_embeds`. ")

        if task_prompt is None and content_prompt is None and prompt_embeds is None:
            raise ValueError("Must provide either `task_prompt` + `content_prompt` or pre-computed `prompt_embeds`. ")

        # Validate prompt types and consistency
        if task_prompt is None:
            raise ValueError("`task_prompt` is missing.")

        if task_prompt is not None and not isinstance(task_prompt, (str, list)):
            raise ValueError(f"`task_prompt` must be str or list, got {type(task_prompt)}")

        if content_prompt is not None and not isinstance(content_prompt, (str, list)):
            raise ValueError(f"`content_prompt` must be str or list, got {type(content_prompt)}")

        if isinstance(task_prompt, list) or isinstance(content_prompt, list):
            if not isinstance(task_prompt, list) or not isinstance(content_prompt, list):
                raise ValueError(
                    f"`task_prompt` and `content_prompt` must both be lists, or both be of type str or None, "
                    f"got {type(task_prompt)} and {type(content_prompt)}"
                )
            if len(content_prompt) != len(task_prompt):
                raise ValueError("`task_prompt` and `content_prompt` must have the same length whe they are lists.")

            for sample in image:
                if not isinstance(sample, list) or not isinstance(sample[0], list):
                    raise ValueError("Each sample in the batch must have a 2D list of images.")
                if len({len(row) for row in sample}) != 1:
                    raise ValueError("Each in-context example and query should contain the same number of images.")
                if not any(img is None for img in sample[-1]):
                    raise ValueError("There are no targets in the query, which should be represented as None.")
                for row in sample[:-1]:
                    if any(img is None for img in row):
                        raise ValueError("Images are missing in in-context examples.")

        # Validate embeddings
        if prompt_embeds is not None and pooled_prompt_embeds is None:
            raise ValueError(
                "If `prompt_embeds` are provided, `pooled_prompt_embeds` also have to be passed. Make sure to generate `pooled_prompt_embeds` from the same text encoder that was used to generate `prompt_embeds`."
            )

        # Validate sequence length
        if max_sequence_length is not None and max_sequence_length > 512:
            raise ValueError(f"max_sequence_length cannot exceed 512, got {max_sequence_length}")


    @staticmethod
    # Copied from diffusers.pipelines.flux.pipeline_flux.FluxPipeline._unpack_latents with _latents->_latents_upsampling
    def _unpack_latents(latents, height, width, vae_scale_factor):
        batch_size, num_patches, channels = latents.shape

        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (vae_scale_factor * 2))
        width = 2 * (int(width) // (vae_scale_factor * 2))

        latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)

        latents = latents.reshape(batch_size, channels // (2 * 2), height, width)

        return latents

    def _prepare_latents(self, image, mask, gen, vae_scale_factor, device, dtype):
        """Helper function to prepare latents for a single batch."""
        # Concatenate images and masks along width dimension
        image = [torch.cat(img, dim=3).to(device=device, dtype=dtype) for img in image]
        mask = [torch.cat(m, dim=3).to(device=device, dtype=dtype) for m in mask]

        # Generate latent image IDs
        latent_image_ids = self._prepare_latent_image_ids(image, vae_scale_factor, device, dtype)

        # For post-upsampling, use zero images for masked latents
        image_latent = [self._encode_vae_image(img, gen) for img in image]
        masked_image_latent = [self._encode_vae_image(img * 0, gen) for img in image]

        for i in range(len(image_latent)):
            # Rearrange latents and masks for patch processing
            num_channels_latents, height, width = image_latent[i].shape[1:]
            image_latent[i] = self._pack_latents(image_latent[i], 1, num_channels_latents, height, width)
            masked_image_latent[i] = self._pack_latents(masked_image_latent[i], 1, num_channels_latents, height, width)

            # Rearrange masks for patch processing
            num_channels_latents, height, width = mask[i].shape[1:]
            mask[i] = mask[i].view(
                1,
                num_channels_latents,
                height // vae_scale_factor,
                vae_scale_factor,
                width // vae_scale_factor,
                vae_scale_factor,
            )
            mask[i] = mask[i].permute(0, 1, 3, 5, 2, 4)
            mask[i] = mask[i].reshape(
                1,
                num_channels_latents * (vae_scale_factor**2),
                height // vae_scale_factor,
                width // vae_scale_factor,
            )
            mask[i] = self._pack_latents(
                mask[i],
                1,
                num_channels_latents * (vae_scale_factor**2),
                height // vae_scale_factor,
                width // vae_scale_factor,
            )

        # Concatenate along batch dimension
        image_latent = torch.cat(image_latent, dim=1)
        masked_image_latent = torch.cat(masked_image_latent, dim=1)
        mask = torch.cat(mask, dim=1)

        return image_latent, masked_image_latent, mask, latent_image_ids

    @torch.no_grad()
    def __call__(
        self,
        task_prompt: Union[str, List[str]] = None,
        content_prompt: Union[str, List[str]] = None,
        image: Optional[torch.FloatTensor] = None,
        upsampling_height: Optional[int] = None,
        upsampling_width: Optional[int] = None,
        num_inference_steps: int = 50,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 30.0,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        upsampling_strength: float = 1.0,
    ):
        r"""
        Function invoked when calling the VisualCloze pipeline for generation.

        Args:
            task_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to define the task intention.
            content_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to define the content or caption of the target image to be generated.
            image (`torch.Tensor`, `PIL.Image.Image`, `np.ndarray`, `List[torch.Tensor]`, `List[PIL.Image.Image]`, or `List[np.ndarray]`):
                `Image`, numpy array or tensor representing an image batch to be used as the starting point. For both
                numpy array and pytorch tensor, the expected value range is between `[0, 1]` If it's a tensor or a list
                or tensors, the expected shape should be `(B, C, H, W)` or `(C, H, W)`. If it is a numpy array or a
                list of arrays, the expected shape should be `(B, H, W, C)` or `(H, W, C)`.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            guidance_scale (`float`, *optional*, defaults to 30.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
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
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.flux.FluxPipelineOutput`] instead of a plain tuple.
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
            max_sequence_length (`int` defaults to 512): Maximum sequence length to use with the `prompt`.

        Examples:

        Returns:
            [`~pipelines.flux.FluxPipelineOutput`] or `tuple`: [`~pipelines.flux.FluxPipelineOutput`] if `return_dict`
            is True, otherwise a `tuple`. When returning a tuple, the first element is a list with the generated
            images.
        """

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            image,
            task_prompt,
            content_prompt,
            upsampling_height=upsampling_height, 
            upsampling_width=upsampling_width,
            strength=upsampling_strength,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        processor_output = self.image_processor.preprocess(
            task_prompt, content_prompt, image, vae_scale_factor=self.vae_scale_factor
        )

        # 2. Define call parameters
        if processor_output["task_prompt"] is not None and isinstance(processor_output["task_prompt"], str):
            batch_size = 1
        elif processor_output["task_prompt"] is not None and isinstance(processor_output["task_prompt"], list):
            batch_size = len(processor_output["task_prompt"])

        device = self._execution_device

         # 3. Prepare prompt embeddings
        lora_scale = (
            self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        )
        prompt_embeds, pooled_prompt_embeds, text_ids = self.encode_prompt(
            layout_prompt=processor_output["layout_prompt"],
            task_prompt=processor_output["task_prompt"],
            content_prompt=processor_output["content_prompt"],
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

        # 4. Prepare timesteps
        # Calculate sequence length and shift factor
        image_seq_len = sum(
            (size[0] // self.vae_scale_factor // 2) * (size[1] // self.vae_scale_factor // 2)
            for sample in processor_output["image_size"][0]
            for size in sample
        )

        # Calculate noise schedule parameters
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )

        # Get timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, upsampling_strength, device)

        if num_inference_steps < 1:
            raise ValueError(
                f"After adjusting the num_inference_steps by strength parameter: {upsampling_strength}, the number of pipeline"
                f"steps is {num_inference_steps} which is < 1 and not appropriate for this pipeline."
            )

        # 5. Prepare latent variables
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)
        latents, masked_image_latents, latent_image_ids = self.prepare_latents(
            processor_output["init_image"],
            processor_output["mask"],
            latent_timestep,
            batch_size * num_images_per_prompt,
            prompt_embeds.dtype,
            device,
            generator,
            vae_scale_factor=self.vae_scale_factor,
        )

        # Calculate warmup steps
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # Prepare guidance
        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        # 6. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # Broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latents.shape[0]).to(latents.dtype)
                latent_model_input = torch.cat((latents, masked_image_latents), dim=2)

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]

                # Compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # Some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                # Call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                # XLA optimization
                if XLA_AVAILABLE:
                    xm.mark_step()

         # 7. Post-process the image
        # Crop the target image
        # Since the generated image is a concatenation of the conditional and target regions,
        # we need to extract only the target regions based on their positions
        image = []
        if output_type == "latent":
            image = latents
        else:
            latents = self._unpack_latents(latents, upsampling_height, upsampling_width, self.vae_scale_factor)
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return FluxPipelineOutput(images=image)
