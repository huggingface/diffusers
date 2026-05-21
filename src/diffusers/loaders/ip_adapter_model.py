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
"""Model-side IP-Adapter machinery.

Generic orchestration (set processors, build ``MultiIPAdapterImageProjection``, flip ``encoder_hid_dim_type``) lives on
:class:`IPAdapterModelMixin`. Per-model conversion lives in an :class:`IPAdapterHandler` declared next to the model
(e.g. ``flux/ip_adapter.py`` exports ``FLUX_IP_ADAPTER``), assigned to the model class as
``_ip_adapter = FLUX_IP_ADAPTER``.
"""

from dataclasses import dataclass
from typing import Callable, Optional

from ..models.embeddings import MultiIPAdapterImageProjection
from ..utils import is_torch_version, logging


_LOW_CPU_MEM_USAGE_DEFAULT = is_torch_version(">=", "1.9.0")


logger = logging.get_logger(__name__)


@dataclass
class IPAdapterHandler:
    """Composition-style holder for a model class's IP-Adapter conversion callables.

    Attached as the ``_ip_adapter`` class attribute on :class:`IPAdapterModelMixin` (overridden per-model). The converter callables receive the
    model instance because they need to read its config (``attn_processors``, ``inner_dim``, etc.).

    Attributes:
        convert_attn_to_diffusers_fn:
            Callable ``(model, state_dicts, low_cpu_mem_usage=False) -> dict[str, AttnProcessor]`` returning the
            attn-processor dict ready for ``set_attn_processor``.
        convert_image_proj_to_diffusers_fn: Callable
            ``(model, image_proj_state_dict, low_cpu_mem_usage=False) -> ImageProjection`` returning the image
            projection module.
    """

    convert_attn_to_diffusers_fn: Optional[Callable] = None
    convert_image_proj_to_diffusers_fn: Optional[Callable] = None

    @property
    def supports_ip_adapter(self) -> bool:
        """Whether the model has both converters registered (required to actually load weights)."""
        return self.convert_attn_to_diffusers_fn is not None and self.convert_image_proj_to_diffusers_fn is not None

    def convert_attn_processors(self, model, state_dicts, low_cpu_mem_usage: bool = False):
        """Build the attention-processor dict for a list of IP-Adapter state dicts.

        Receives the model so the converter can inspect ``model.attn_processors``, ``model.config``,
        ``model.inner_dim``, etc.
        """
        if self.convert_attn_to_diffusers_fn is None:
            raise NotImplementedError(
                f"{type(model).__name__} did not register `convert_attn_to_diffusers_fn` in its IPAdapterHandler."
            )
        return self.convert_attn_to_diffusers_fn(model, state_dicts, low_cpu_mem_usage=low_cpu_mem_usage)

    def convert_image_proj(self, model, image_proj_state_dict, low_cpu_mem_usage: bool = False):
        """Build the image-projection module from a single IP-Adapter state dict."""
        if self.convert_image_proj_to_diffusers_fn is None:
            raise NotImplementedError(
                f"{type(model).__name__} did not register `convert_image_proj_to_diffusers_fn` in its "
                f"IPAdapterHandler."
            )
        return self.convert_image_proj_to_diffusers_fn(
            model, image_proj_state_dict, low_cpu_mem_usage=low_cpu_mem_usage
        )


class IPAdapterModelMixin:
    """Generic IP-Adapter loader for diffusers transformer / UNet models.

    The per-model conversion callables live on ``self._ip_adapter`` (an :class:`IPAdapterHandler` assigned as a class
    attribute by the model). This mixin owns only the orchestration: dispatching to the converters, wiring up
    ``MultiIPAdapterImageProjection``, and flipping ``encoder_hid_dim_type``.
    """

    # Per-model IP-Adapter conversion config. Defaults to an empty handler; models that support IP-Adapter assign
    # ``_ip_adapter = FLUX_IP_ADAPTER`` (etc.) in their class body. Calling ``_load_ip_adapter_weights`` on a
    # model that didn't override raises ``NotImplementedError`` from inside the handler.
    _ip_adapter: IPAdapterHandler = IPAdapterHandler()

    @classmethod
    def _metadata(cls):
        """Contribute the ``ip_adapter`` row to :class:`ModelMetadata` when converters are registered."""
        from ..models.modeling_utils import DOCS_BASE

        if not cls._ip_adapter.supports_ip_adapter:
            return {}
        return {
            "ip_adapter": (
                "yes",
                "Supports loading IP-Adapter weights (image-conditioning adapters).",
                f"{DOCS_BASE}/using-diffusers/ip_adapter",
            )
        }

    def _load_ip_adapter_weights(self, state_dicts, low_cpu_mem_usage=_LOW_CPU_MEM_USAGE_DEFAULT):
        """Install IP-Adapter weights on the model.

        ``state_dicts`` is a single state dict (or a list, for multi-adapter loading); each dict must contain
        ``"image_proj"`` and ``"ip_adapter"`` sub-dicts.
        """
        if not self._ip_adapter.supports_ip_adapter:
            raise NotImplementedError(
                f"{type(self).__name__} did not register IP-Adapter converters in its IPAdapterHandler."
            )

        if not isinstance(state_dicts, list):
            state_dicts = [state_dicts]

        self.encoder_hid_proj = None

        attn_procs = self._ip_adapter.convert_attn_processors(
            self, state_dicts, low_cpu_mem_usage=low_cpu_mem_usage
        )
        self.set_attn_processor(attn_procs)

        image_projection_layers = []
        for state_dict in state_dicts:
            image_projection_layer = self._ip_adapter.convert_image_proj(
                self, state_dict["image_proj"], low_cpu_mem_usage=low_cpu_mem_usage
            )
            image_projection_layers.append(image_projection_layer)

        self.encoder_hid_proj = MultiIPAdapterImageProjection(image_projection_layers)
        self.config.encoder_hid_dim_type = "ip_image_proj"
