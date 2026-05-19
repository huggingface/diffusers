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
"""Model-side IP-Adapter mixin.

Generic orchestration (set processors, build ``MultiIPAdapterImageProjection``, flip ``encoder_hid_dim_type``) lives
here. Per-model conversion lives in a ``IPAdapterMetadata`` declared next to the model — e.g. ``flux/ip_adapter.py``
exports ``FLUX_IP_ADAPTER_METADATA``, which is then composed into the model's ``ModelMetadata`` and attached via
``@register_model_metadata``.
"""

from ..models.embeddings import MultiIPAdapterImageProjection
from ..models.modeling_utils import _LOW_CPU_MEM_USAGE_DEFAULT
from ..utils import logging


logger = logging.get_logger(__name__)


class IPAdapterModelMixin:
    """Metadata-driven IP-Adapter loader for diffusers transformer / UNet models.

    Reads the per-model converters from ``IPAdapterMetadata`` (mirrored onto
    ``cls._convert_ip_adapter_attn_to_diffusers`` and ``cls._convert_ip_adapter_image_proj_to_diffusers`` by
    ``register_model_metadata``).
    """

    # No-op defaults; populated per-model by ``register_model_metadata``.
    _convert_ip_adapter_attn_to_diffusers = None
    _convert_ip_adapter_image_proj_to_diffusers = None

    def _load_ip_adapter_weights(self, state_dicts, low_cpu_mem_usage=_LOW_CPU_MEM_USAGE_DEFAULT):
        """Install IP-Adapter weights on the model.

        ``state_dicts`` is a single state dict (or a list, for multi-adapter loading); each dict must contain
        ``"image_proj"`` and ``"ip_adapter"`` sub-dicts.
        """
        if (
            self._convert_ip_adapter_attn_to_diffusers is None
            or self._convert_ip_adapter_image_proj_to_diffusers is None
        ):
            raise NotImplementedError(
                f"{type(self).__name__} did not register IP-Adapter converters in its IPAdapterMetadata."
            )

        if not isinstance(state_dicts, list):
            state_dicts = [state_dicts]

        self.encoder_hid_proj = None

        attn_procs = self._convert_ip_adapter_attn_to_diffusers(self, state_dicts, low_cpu_mem_usage=low_cpu_mem_usage)
        self.set_attn_processor(attn_procs)

        image_projection_layers = []
        for state_dict in state_dicts:
            image_projection_layer = self._convert_ip_adapter_image_proj_to_diffusers(
                self, state_dict["image_proj"], low_cpu_mem_usage=low_cpu_mem_usage
            )
            image_projection_layers.append(image_projection_layer)

        self.encoder_hid_proj = MultiIPAdapterImageProjection(image_projection_layers)
        self.config.encoder_hid_dim_type = "ip_image_proj"
