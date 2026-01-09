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

from typing import Any, Callable, Dict, List, Optional, Union

import torch
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

from ...loaders import FluxLoraLoaderMixin, FromSingleFileMixin, TextualInversionLoaderMixin
from ...models.autoencoders import AutoencoderKL
from ...models.transformers import FluxTransformer2DModel
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import is_torch_xla_available, logging, replace_example_docstring
from ..flux.pipeline_flux_fill import FluxFillPipeline as VisualClozeUpsamplingPipeline
from ..flux.pipeline_output import FluxPipelineOutput
from ..pipeline_utils import DiffusionPipeline
from .pipeline_visualcloze_generation import VisualClozeGenerationPipeline


if is_torch_xla_available():
    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```python
        >>> import torch
        >>> from diffusers import VisualClozePipeline
        >>> from diffusers.utils import load_image

        >>> image_paths = [
        ...     # in-context examples
        ...     [
        ...         load_image(
        ...             "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/visualcloze/visualcloze_mask2image_incontext-example-1_mask.jpg"
        ...         ),
        ...         load_image(
        ...             "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/visualcloze/visualcloze_mask2image_incontext-example-1_image.jpg"
        ...         ),
        ...     ],
        ...     # query with the target image
        ...     [
        ...         load_image(
        ...             "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/visualcloze/visualcloze_mask2image_query_mask.jpg"
        ...         ),
        ...         None,  # No image needed for the target image
        ...     ],
        ... ]
        >>> task_prompt = "In each row, a logical task is demonstrated to achieve [IMAGE2] an aesthetically pleasing photograph based on [IMAGE1] sam 2-generated masks with rich color coding."
        >>> content_prompt = "Majestic photo of a golden eagle perched on a rocky outcrop in a mountainous landscape. The eagle is positioned in the right foreground, facing left, with its sharp beak and keen eyes prominently visible. Its plumage is a mix of dark brown and golden hues, with intricate feather details. The background features a soft-focus view of snow-capped mountains under a cloudy sky, creating a serene and grandiose atmosphere. The foreground includes rugged rocks and patches of green moss. Photorealistic, medium depth of field, soft natural lighting, cool color palette, high contrast, sharp focus on the eagle, blurred background, tranquil, majestic, wildlife photography."
        >>> pipe = VisualClozePipeline.from_pretrained(
        ...     "VisualCloze/VisualClozePipeline-384", resolution=384, torch_dtype=torch.bfloat16
        ... )
        >>> pipe.to("cuda")

        >>> image = pipe(
        ...     task_prompt=task_prompt,
        ...     content_prompt=content_prompt,
        ...     image=image_paths,
        ...     upsampling_width=1344,
        ...     upsampling_height=768,
        ...     upsampling_strength=0.4,
        ...     guidance_scale=30,
        ...     num_inference_steps=30,
        ...     max_sequence_length=512,
        ...     generator=torch.Generator("cpu").manual_seed(0),
        ... ).images[0][0]
        >>> image.save("visualcloze.png")
        ```
"""


class VisualClozePipeline(
    DiffusionPipeline,
    FluxLoraLoaderMixin,
    FromSingleFileMixin,
    TextualInversionLoaderMixin,
):
    r"""
    The VisualCloze pipeline for image generation with visual context. Reference:
    https://github.com/lzyhha/VisualCloze/tree/main. This pipeline is designed to generate images based on visual
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

    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        text_encoder_2: T5EncoderModel,
        tokenizer_2: T5TokenizerFast,
        transformer: FluxTransformer2DModel,
        resolution: int = 384,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            transformer=transformer,
            scheduler=scheduler,
        )

        self.generation_pipe = VisualClozeGenerationPipeline(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            transformer=transformer,
            scheduler=scheduler,
            resolution=resolution,
        )
        self.upsampling_pipe = VisualClozeUpsamplingPipeline(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            transformer=transformer,
            scheduler=scheduler,
        )

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

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
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
            upsampling_height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image (i.e., output image) after upsampling via SDEdit. By
                default, the image is upsampled by a factor of three, and the base resolution is determined by the
                resolution parameter of the pipeline. When only one of `upsampling_height` or `upsampling_width` is
                specified, the other will be automatically set based on the aspect ratio.
            upsampling_width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image (i.e., output image) after upsampling via SDEdit. By
                default, the image is upsampled by a factor of three, and the base resolution is determined by the
                resolution parameter of the pipeline. When only one of `upsampling_height` or `upsampling_width` is
                specified, the other will be automatically set based on the aspect ratio.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            guidance_scale (`float`, *optional*, defaults to 30.0):
                Guidance scale as defined in [Classifier-Free Diffusion
                Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2.
                of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting
                `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to
                the text `prompt`, usually at the expense of lower image quality.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will be generated by sampling using the supplied random `generator`.
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
            upsampling_strength (`float`, *optional*, defaults to 1.0):
                Indicates extent to transform the reference `image` when upsampling the results. Must be between 0 and
                1. The generated image is used as a starting point and more noise is added the higher the
                `upsampling_strength`. The number of denoising steps depends on the amount of noise initially added.
                When `upsampling_strength` is 1, added noise is maximum and the denoising process runs for the full
                number of iterations specified in `num_inference_steps`. A value of 0 skips the upsampling step and
                output the results at the resolution of `self.resolution`.

        Examples:

        Returns:
            [`~pipelines.flux.FluxPipelineOutput`] or `tuple`: [`~pipelines.flux.FluxPipelineOutput`] if `return_dict`
            is True, otherwise a `tuple`. When returning a tuple, the first element is a list with the generated
            images.
        """

        generation_output = self.generation_pipe(
            task_prompt=task_prompt,
            content_prompt=content_prompt,
            image=image,
            num_inference_steps=num_inference_steps,
            sigmas=sigmas,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
            generator=generator,
            latents=latents,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            joint_attention_kwargs=joint_attention_kwargs,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
            output_type=output_type if upsampling_strength == 0 else "pil",
        )
        if upsampling_strength == 0:
            if not return_dict:
                return (generation_output,)

            return FluxPipelineOutput(images=generation_output)

        # Upsampling the generated images
        # 1. Prepare the input images and prompts
        if not isinstance(content_prompt, (list)):
            content_prompt = [content_prompt]
        n_target_per_sample = []
        upsampling_image = []
        upsampling_mask = []
        upsampling_prompt = []
        upsampling_generator = generator if isinstance(generator, (torch.Generator,)) else []
        for i in range(len(generation_output.images)):
            n_target_per_sample.append(len(generation_output.images[i]))
            for image in generation_output.images[i]:
                upsampling_image.append(image)
                upsampling_mask.append(Image.new("RGB", image.size, (255, 255, 255)))
                upsampling_prompt.append(
                    content_prompt[i % len(content_prompt)] if content_prompt[i % len(content_prompt)] else ""
                )
                if not isinstance(generator, (torch.Generator,)):
                    upsampling_generator.append(generator[i % len(content_prompt)])

        # 2. Apply the denosing loop
        upsampling_output = self.upsampling_pipe(
            prompt=upsampling_prompt,
            image=upsampling_image,
            mask_image=upsampling_mask,
            height=upsampling_height,
            width=upsampling_width,
            strength=upsampling_strength,
            num_inference_steps=num_inference_steps,
            sigmas=sigmas,
            guidance_scale=guidance_scale,
            generator=upsampling_generator,
            output_type=output_type,
            joint_attention_kwargs=joint_attention_kwargs,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )
        image = upsampling_output.images

        output = []
        if output_type == "pil":
            # Each sample in the batch may have multiple output images. When returning as PIL images,
            # these images cannot be concatenated. Therefore, for each sample,
            # a list is used to represent all the output images.
            output = []
            start = 0
            for n in n_target_per_sample:
                output.append(image[start : start + n])
                start += n
        else:
            output = image

        if not return_dict:
            return (output,)

        return FluxPipelineOutput(images=output)
