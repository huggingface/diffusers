# Copyright 2025 Black Forest Labs and The HuggingFace Team. All rights reserved.
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


from typing import List, Optional, Union

import torch
from PIL import Image
from transformers import (
    CLIPTextModel,
    CLIPTokenizer,
    SiglipImageProcessor,
    SiglipVisionModel,
    T5EncoderModel,
    T5TokenizerFast,
)

from ...image_processor import PipelineImageInput
from ...utils import (
    is_torch_xla_available,
    logging,
    replace_example_docstring,
)
from ..pipeline_utils import DiffusionPipeline
from .modeling_flux import ReduxImageEncoder
from .pipeline_flux_utils import FluxMixin
from .pipeline_output import FluxPriorReduxPipelineOutput


if is_torch_xla_available():
    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import FluxPriorReduxPipeline, FluxPipeline
        >>> from diffusers.utils import load_image

        >>> device = "cuda"
        >>> dtype = torch.bfloat16

        >>> repo_redux = "black-forest-labs/FLUX.1-Redux-dev"
        >>> repo_base = "black-forest-labs/FLUX.1-dev"
        >>> pipe_prior_redux = FluxPriorReduxPipeline.from_pretrained(repo_redux, torch_dtype=dtype).to(device)
        >>> pipe = FluxPipeline.from_pretrained(
        ...     repo_base, text_encoder=None, text_encoder_2=None, torch_dtype=torch.bfloat16
        ... ).to(device)

        >>> image = load_image(
        ...     "https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/style_ziggy/img5.png"
        ... )
        >>> pipe_prior_output = pipe_prior_redux(image)
        >>> images = pipe(
        ...     guidance_scale=2.5,
        ...     num_inference_steps=50,
        ...     generator=torch.Generator("cpu").manual_seed(0),
        ...     **pipe_prior_output,
        ... ).images
        >>> images[0].save("flux-redux.png")
        ```
"""


class FluxPriorReduxPipeline(DiffusionPipeline, FluxMixin):
    r"""
    The Flux Redux pipeline for image-to-image generation.

    Reference: https://blackforestlabs.ai/flux-1-tools/

    Args:
        image_encoder ([`SiglipVisionModel`]):
            SIGLIP vision model to encode the input image.
        feature_extractor ([`SiglipImageProcessor`]):
            Image processor for preprocessing images for the SIGLIP model.
        image_embedder ([`ReduxImageEncoder`]):
            Redux image encoder to process the SIGLIP embeddings.
        text_encoder ([`CLIPTextModel`], *optional*):
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        text_encoder_2 ([`T5EncoderModel`], *optional*):
            [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5EncoderModel), specifically
            the [google/t5-v1_1-xxl](https://huggingface.co/google/t5-v1_1-xxl) variant.
        tokenizer (`CLIPTokenizer`, *optional*):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/en/model_doc/clip#transformers.CLIPTokenizer).
        tokenizer_2 (`T5TokenizerFast`, *optional*):
            Second Tokenizer of class
            [T5TokenizerFast](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5TokenizerFast).
    """

    model_cpu_offload_seq = "image_encoder->image_embedder"
    _optional_components = [
        "text_encoder",
        "tokenizer",
        "text_encoder_2",
        "tokenizer_2",
    ]
    _callback_tensor_inputs = []

    def __init__(
        self,
        image_encoder: SiglipVisionModel,
        feature_extractor: SiglipImageProcessor,
        image_embedder: ReduxImageEncoder,
        text_encoder: CLIPTextModel = None,
        tokenizer: CLIPTokenizer = None,
        text_encoder_2: T5EncoderModel = None,
        tokenizer_2: T5TokenizerFast = None,
    ):
        super().__init__()

        self.register_modules(
            image_encoder=image_encoder,
            feature_extractor=feature_extractor,
            image_embedder=image_embedder,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_encoder_2=text_encoder_2,
            tokenizer_2=tokenizer_2,
        )
        self.tokenizer_max_length = (
            self.tokenizer.model_max_length if hasattr(self, "tokenizer") and self.tokenizer is not None else 77
        )

    def check_inputs(
        self,
        image,
        prompt,
        prompt_2,
        prompt_embeds=None,
        pooled_prompt_embeds=None,
        prompt_embeds_scale=1.0,
        pooled_prompt_embeds_scale=1.0,
    ):
        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt_2 is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt_2`: {prompt_2} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        elif prompt_2 is not None and (not isinstance(prompt_2, str) and not isinstance(prompt_2, list)):
            raise ValueError(f"`prompt_2` has to be of type `str` or `list` but is {type(prompt_2)}")
        if prompt is not None and (isinstance(prompt, list) and isinstance(image, list) and len(prompt) != len(image)):
            raise ValueError(
                f"number of prompts must be equal to number of images, but {len(prompt)} prompts were provided and {len(image)} images"
            )
        if prompt_embeds is not None and pooled_prompt_embeds is None:
            raise ValueError(
                "If `prompt_embeds` are provided, `pooled_prompt_embeds` also have to be passed. Make sure to generate `pooled_prompt_embeds` from the same text encoder that was used to generate `prompt_embeds`."
            )
        if isinstance(prompt_embeds_scale, list) and (
            isinstance(image, list) and len(prompt_embeds_scale) != len(image)
        ):
            raise ValueError(
                f"number of weights must be equal to number of images, but {len(prompt_embeds_scale)} weights were provided and {len(image)} images"
            )

    def encode_image(self, image, device, num_images_per_prompt):
        dtype = next(self.image_encoder.parameters()).dtype
        image = self.feature_extractor.preprocess(
            images=image, do_resize=True, return_tensors="pt", do_convert_rgb=True
        )
        image = image.to(device=device, dtype=dtype)

        image_enc_hidden_states = self.image_encoder(**image).last_hidden_state
        image_enc_hidden_states = image_enc_hidden_states.repeat_interleave(num_images_per_prompt, dim=0)

        return image_enc_hidden_states

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        image: PipelineImageInput,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        prompt_embeds_scale: Optional[Union[float, List[float]]] = 1.0,
        pooled_prompt_embeds_scale: Optional[Union[float, List[float]]] = 1.0,
        return_dict: bool = True,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            image (`torch.Tensor`, `PIL.Image.Image`, `np.ndarray`, `List[torch.Tensor]`, `List[PIL.Image.Image]`, or `List[np.ndarray]`):
                `Image`, numpy array or tensor representing an image batch to be used as the starting point. For both
                numpy array and pytorch tensor, the expected value range is between `[0, 1]` If it's a tensor or a list
                or tensors, the expected shape should be `(B, C, H, W)` or `(C, H, W)`. If it is a numpy array or a
                list of arrays, the expected shape should be `(B, H, W, C)` or `(H, W, C)`
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. **experimental feature**: to use this feature,
                make sure to explicitly load text encoders to the pipeline. Prompts will be ignored if text encoders
                are not loaded.
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to the `tokenizer_2` and `text_encoder_2`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.flux.FluxPriorReduxPipelineOutput`] instead of a plain tuple.

        Examples:

        Returns:
            [`~pipelines.flux.FluxPriorReduxPipelineOutput`] or `tuple`:
            [`~pipelines.flux.FluxPriorReduxPipelineOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is a list with the generated images.
        """

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            image,
            prompt,
            prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            prompt_embeds_scale=prompt_embeds_scale,
            pooled_prompt_embeds_scale=pooled_prompt_embeds_scale,
        )

        # 2. Define call parameters
        if image is not None and isinstance(image, Image.Image):
            batch_size = 1
        elif image is not None and isinstance(image, list):
            batch_size = len(image)
        else:
            batch_size = image.shape[0]
        if prompt is not None and isinstance(prompt, str):
            prompt = batch_size * [prompt]
        if isinstance(prompt_embeds_scale, float):
            prompt_embeds_scale = batch_size * [prompt_embeds_scale]
        if isinstance(pooled_prompt_embeds_scale, float):
            pooled_prompt_embeds_scale = batch_size * [pooled_prompt_embeds_scale]

        device = self._execution_device

        # 3. Prepare image embeddings
        image_latents = self.encode_image(image, device, 1)

        image_embeds = self.image_embedder(image_latents).image_embeds
        image_embeds = image_embeds.to(device=device)

        # 3. Prepare (dummy) text embeddings
        if hasattr(self, "text_encoder") and self.text_encoder is not None:
            (
                prompt_embeds,
                pooled_prompt_embeds,
                _,
            ) = self.encode_prompt(
                prompt=prompt,
                prompt_2=prompt_2,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                device=device,
                num_images_per_prompt=1,
                max_sequence_length=512,
                lora_scale=None,
            )
        else:
            if prompt is not None:
                logger.warning(
                    "prompt input is ignored when text encoders are not loaded to the pipeline. "
                    "Make sure to explicitly load the text encoders to enable prompt input. "
                )
            # max_sequence_length is 512, t5 encoder hidden size is 4096
            prompt_embeds = torch.zeros((batch_size, 512, 4096), device=device, dtype=image_embeds.dtype)
            # pooled_prompt_embeds is 768, clip text encoder hidden size
            pooled_prompt_embeds = torch.zeros((batch_size, 768), device=device, dtype=image_embeds.dtype)

        # scale & concatenate image and text embeddings
        prompt_embeds = torch.cat([prompt_embeds, image_embeds], dim=1)

        prompt_embeds *= torch.tensor(prompt_embeds_scale, device=device, dtype=image_embeds.dtype)[:, None, None]
        pooled_prompt_embeds *= torch.tensor(pooled_prompt_embeds_scale, device=device, dtype=image_embeds.dtype)[
            :, None
        ]

        # weighted sum
        prompt_embeds = torch.sum(prompt_embeds, dim=0, keepdim=True)
        pooled_prompt_embeds = torch.sum(pooled_prompt_embeds, dim=0, keepdim=True)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (prompt_embeds, pooled_prompt_embeds)

        return FluxPriorReduxPipelineOutput(prompt_embeds=prompt_embeds, pooled_prompt_embeds=pooled_prompt_embeds)
