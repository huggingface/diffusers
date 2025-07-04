# Copyright 2025 The HuggingFace Team. All rights reserved.
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

from ..hooks import LayerSkipConfig
from .skip_layer_guidance import SkipLayerGuidance


class PerturbedAttentionGuidance(SkipLayerGuidance):
    """
    Perturbed Attention Guidance (PAG): https://huggingface.co/papers/2403.17377

    The intution behind PAG can be thought of as moving the CFG predicted distribution estimates further away from
    worse versions of the conditional distribution estimates. PAG was one of the first techniques to introduce the idea
    of using a worse version of the trained model for better guiding itself in the denoising process. It perturbs the
    attention scores of the latent stream by replacing the score matrix with an identity matrix for selectively chosen
    layers.

    Additional reading:
    - [Guiding a Diffusion Model with a Bad Version of Itself](https://huggingface.co/papers/2406.02507)

    PAG is implemented as a specialization of the SkipLayerGuidance implementation due to similarities in the
    configuration parameters and implementation details. However, it should be noted that PAG was published prior to
    SLG becoming

    Args:
        guidance_scale (`float`, defaults to `7.5`):
            The scale parameter for classifier-free guidance. Higher values result in stronger conditioning on the text
            prompt, while lower values allow for more freedom in generation. Higher values may lead to saturation and
            deterioration of image quality.
        perturbed_guidance_scale (`float`, defaults to `2.8`):
            The scale parameter for perturbed attention guidance.
        perturbed_guidance_start (`float`, defaults to `0.01`):
            The fraction of the total number of denoising steps after which perturbed attention guidance starts.
        perturbed_guidance_stop (`float`, defaults to `0.2`):
            The fraction of the total number of denoising steps after which perturbed attention guidance stops.
        perturbed_guidance_layers (`int` or `List[int]`, *optional*):
            The layer indices to apply perturbed attention guidance to. Can be a single integer or a list of integers.
            If not provided, `skip_layer_config` must be provided.
        skip_layer_config (`LayerSkipConfig` or `List[LayerSkipConfig]`, *optional*):
            The configuration for the perturbed attention guidance. Can be a single `LayerSkipConfig` or a list of
            `LayerSkipConfig`. If not provided, `perturbed_guidance_layers` must be provided.
        guidance_rescale (`float`, defaults to `0.0`):
            The rescale factor applied to the noise predictions. This is used to improve image quality and fix
            overexposure. Based on Section 3.4 from [Common Diffusion Noise Schedules and Sample Steps are
            Flawed](https://huggingface.co/papers/2305.08891).
        use_original_formulation (`bool`, defaults to `False`):
            Whether to use the original formulation of classifier-free guidance as proposed in the paper. By default,
            we use the diffusers-native implementation that has been in the codebase for a long time. See
            [~guiders.classifier_free_guidance.ClassifierFreeGuidance] for more details.
        start (`float`, defaults to `0.01`):
            The fraction of the total number of denoising steps after which guidance starts.
        stop (`float`, defaults to `0.2`):
            The fraction of the total number of denoising steps after which guidance stops.
    """

    # NOTE: The current implementation does not account for joint latent conditioning (text + image/video tokens in
    # the same latent stream). It assumes the entire latent is a single stream of visual tokens. It would be very
    # complex to support joint latent conditioning in a model-agnostic manner without specializing the implementation
    # for each model architecture.

    def __init__(
        guidance_scale: float = 7.5,
        perturbed_guidance_scale: float = 2.8,
        perturbed_guidance_start: float = 0.01,
        perturbed_guidance_stop: float = 0.2,
        perturbed_guidance_layers: Optional[Union[int, List[int]]] = None,
        skip_layer_config: Union[LayerSkipConfig, List[LayerSkipConfig]] = None,
        guidance_rescale: float = 0.0,
        use_original_formulation: bool = False,
        start: float = 0.0,
        stop: float = 1.0,
    ):
        if skip_layer_config is None:
            if perturbed_guidance_layers is None:
                raise ValueError(
                    "`perturbed_guidance_layers` must be provided if `skip_layer_config` is not specified."
                )
            skip_layer_config = LayerSkipConfig(
                indices=perturbed_guidance_layers,
                skip_attention=False,
                skip_attention_scores=True,
                skip_ff=False,
            )
        else:
            if perturbed_guidance_layers is not None:
                raise ValueError(
                    "`perturbed_guidance_layers` should not be provided if `skip_layer_config` is specified."
                )

        super().__init__(
            guidance_scale=guidance_scale,
            skip_layer_guidance_scale=perturbed_guidance_scale,
            skip_layer_guidance_start=perturbed_guidance_start,
            skip_layer_guidance_stop=perturbed_guidance_stop,
            skip_layer_guidance_layers=perturbed_guidance_layers,
            guidance_rescale=guidance_rescale,
            use_original_formulation=use_original_formulation,
            start=start,
            stop=stop,
        )
