# Copyright 2026 The HuggingFace Team. All rights reserved.
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

from ...utils import is_pan2_guardrail_available, logging
from ..modular_pipeline import ModularPipeline


if is_pan2_guardrail_available():
    from pan2_guardrail import PAN2SafetyChecker
else:

    class PAN2SafetyChecker:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "`pan2_guardrail` is not installed. Please install it to use the safety checker for PAN2: "
                "`pip install git+https://github.com/MBZUAI-IFM/PAN2-guardrail`."
            )

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class PAN2ModularPipeline(ModularPipeline):
    """
    A ModularPipeline for the video generator of PAN2: text-to-video, and image-to-video from a first frame.

    ```py
    import torch
    from diffusers import ModularPipeline

    pipe = ModularPipeline.from_pretrained("IFM/PAN2")
    pipe.load_components(dtype=torch.bfloat16)
    pipe.enable_safety_checker()
    ```
    """

    default_blocks_name = "PAN2Blocks"

    @property
    def vae_scale_factor_spatial(self):
        if getattr(self, "vae", None) is not None:
            return self.vae.config.scale_factor_spatial
        return 16

    @property
    def vae_scale_factor_temporal(self):
        if getattr(self, "vae", None) is not None:
            return self.vae.config.scale_factor_temporal
        return 4

    @property
    def num_channels_latents(self):
        if getattr(self, "vae", None) is not None:
            return self.vae.config.z_dim
        return 48

    @property
    def requires_unconditional_embeds(self):
        if getattr(self, "guider", None) is not None:
            return self.guider._enabled and self.guider.num_conditions > 1
        return False

    def enable_safety_checker(self, safety_checker=None):
        r"""
        Check the prompt and the generated videos with `safety_checker`, or with the PAN2 guardrail if none is given
        and none is loaded yet.
        """
        if safety_checker is not None:
            self.safety_checker = safety_checker
        elif getattr(self, "safety_checker", None) is None:
            self.safety_checker = PAN2SafetyChecker.from_pretrained()
        self._is_safety_checker_enabled = True

    def disable_safety_checker(self):
        self._is_safety_checker_enabled = False

    @property
    def requires_safety_checker(self):
        return getattr(self, "_is_safety_checker_enabled", self.config.enable_safety_checker)

    @property
    def system_message(self):
        r"""The system message of the chat template the prompt is encoded with."""
        return "You are a helpful assistant. Generate the video with the following instructions."

    @property
    def text_encoder_num_hidden_layers_to_skip(self):
        r"""PAN2 conditions on `hidden_states[-3]` of the text encoder, i.e. it skips the last two layers."""
        return 2
