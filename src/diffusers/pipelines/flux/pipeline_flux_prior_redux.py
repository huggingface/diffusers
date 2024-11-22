# Copyright 2024 Black Forest Labs and The HuggingFace Team. All rights reserved.
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


import torch
from transformers import SiglipImageProcessor, SiglipVisionModel

from ...image_processor import PipelineImageInput
from ...utils import (
    is_torch_xla_available,
    logging,
    replace_example_docstring,
)
from ..pipeline_utils import DiffusionPipeline
from .modeling_flux import ReduxImageEncoder
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
        >>> from diffusers import FluxPipeline

        >>> pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
        >>> pipe.to("cuda")
        >>> prompt = "A cat holding a sign that says hello world"
        >>> # Depending on the variant being used, the pipeline call will slightly vary.
        >>> # Refer to the pipeline documentation for more details.
        >>> image = pipe(prompt, num_inference_steps=4, guidance_scale=0.0).images[0]
        >>> image.save("flux.png")
        ```
"""


class FluxPriorReduxPipeline(DiffusionPipeline):
    r"""
    The Flux pipeline for text-to-image generation.

    Reference: https://blackforestlabs.ai/announcing-black-forest-labs/

    Args:
        transformer ([`FluxTransformer2DModel`]):
            Conditional Transformer (MMDiT) architecture to denoise the encoded image latents.
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
    """

    model_cpu_offload_seq = "image_encoder->image_embedder"
    _optional_components = []
    _callback_tensor_inputs = []

    def __init__(
        self,
        image_encoder: SiglipVisionModel,
        feature_extractor: SiglipImageProcessor,
        image_embedder: ReduxImageEncoder,
    ):
        super().__init__()

        self.register_modules(
            image_encoder=image_encoder,
            feature_extractor=feature_extractor,
            image_embedder=image_embedder,
        )

    def encode_image(self, image, device, num_images_per_prompt):
        dtype = next(self.image_encoder.parameters()).dtype
        image = self.feature_extractor.preprocess(
            images=[image], do_resize=True, return_tensors="pt", do_convert_rgb=True
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
        return_dict: bool = True,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.

        Examples:

        Returns:
            [`~pipelines.flux.FluxPipelineOutput`] or `tuple`: [`~pipelines.flux.FluxPipelineOutput`] if `return_dict`
            is True, otherwise a `tuple`. When returning a tuple, the first element is a list with the generated
            images.
        """

        # 2. Define call parameters
        device = self._execution_device

        image_latents = self.encode_image(image, device, 1)
        image_embeds = self.image_embedder(image_latents).image_embeds

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image_embeds,)

        return FluxPriorReduxPipelineOutput(image_embeds=image_embeds)
