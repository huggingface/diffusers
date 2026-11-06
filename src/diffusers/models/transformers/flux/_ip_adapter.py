# Copyright 2025 Black Forest Labs, The HuggingFace Team. All rights reserved.
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
"""Flux-specific IP-Adapter loading.

IP-Adapter behavior — what's in the state dict, what the attn processors look like, which blocks they bind to — varies
enough across models that a generic mixin can't really capture the orchestration. Flux owns its own
``_load_ip_adapter_weights`` here, including the loop over blocks, the choice to skip single-stream blocks, and the
projection-dim computation.

``FluxIPAdapterMixin`` is added to ``FluxTransformer2DModel``'s bases in ``flux/model.py``. Models that don't support
IP-Adapter simply don't inherit anything — there's no opt-in handler default to override.
"""

from contextlib import nullcontext

from ....models.embeddings import ImageProjection, MultiIPAdapterImageProjection
from ....models.model_loading_utils import load_model_dict_into_meta
from ....models.modeling_utils import _LOW_CPU_MEM_USAGE_DEFAULT, DOCS_BASE
from ....utils import is_accelerate_available, is_torch_version, logging
from ....utils.torch_utils import empty_device_cache


logger = logging.get_logger(__name__)


def _resolve_init_context(low_cpu_mem_usage):
    """Return ``(init_context, low_cpu_mem_usage)`` — disables low-cpu init if accelerate is missing."""
    if low_cpu_mem_usage:
        if is_accelerate_available():
            from accelerate import init_empty_weights

            if not is_torch_version(">=", "1.9.0"):
                raise NotImplementedError(
                    "Low memory initialization requires torch >= 1.9.0. Please either update your PyTorch "
                    "version or set `low_cpu_mem_usage=False`."
                )
            return init_empty_weights, True

        logger.warning(
            "Cannot initialize model with low cpu memory usage because `accelerate` was not found in the "
            "environment. Defaulting to `low_cpu_mem_usage=False`. It is strongly recommended to install "
            "`accelerate` for faster and less memory-intense model loading. You can do so with: \n```\npip "
            "install accelerate\n```\n."
        )
    return nullcontext, False


def _convert_image_proj(model, state_dict, low_cpu_mem_usage=_LOW_CPU_MEM_USAGE_DEFAULT):
    """Build a Flux ``ImageProjection`` from an IP-Adapter ``image_proj`` state dict."""
    init_context, low_cpu_mem_usage = _resolve_init_context(low_cpu_mem_usage)

    # ``proj.weight`` rows == cross_attention_dim * num_image_text_embeds. The two
    # supported configurations: 4 tokens (default) and 16 tokens (when rows == 65536).
    num_image_text_embeds = 16 if state_dict["proj.weight"].shape[0] == 65536 else 4
    clip_embeddings_dim = state_dict["proj.weight"].shape[-1]
    cross_attention_dim = state_dict["proj.weight"].shape[0] // num_image_text_embeds

    with init_context():
        image_projection = ImageProjection(
            cross_attention_dim=cross_attention_dim,
            image_embed_dim=clip_embeddings_dim,
            num_image_text_embeds=num_image_text_embeds,
        )

    updated_state_dict = {key.replace("proj", "image_embeds"): value for key, value in state_dict.items()}

    if low_cpu_mem_usage:
        load_model_dict_into_meta(
            image_projection, updated_state_dict, device_map={"": model.device}, dtype=model.dtype
        )
        empty_device_cache()
    else:
        image_projection.load_state_dict(updated_state_dict, strict=True)

    return image_projection


def _convert_attn_processors(model, state_dicts, low_cpu_mem_usage=_LOW_CPU_MEM_USAGE_DEFAULT):
    """Build the IP-Adapter attn-processor dict for a ``FluxTransformer2DModel``.

    Single-stream blocks keep their existing processor; double-stream blocks get a ``FluxIPAdapterAttnProcessor``
    loaded with the per-state-dict ``to_k_ip`` / ``to_v_ip`` weights.
    """
    from .model import FluxIPAdapterAttnProcessor

    init_context, low_cpu_mem_usage = _resolve_init_context(low_cpu_mem_usage)

    attn_procs = {}
    key_id = 0
    for name in model.attn_processors:
        if name.startswith("single_transformer_blocks"):
            attn_procs[name] = model.attn_processors[name].__class__()
            continue

        num_image_text_embeds = [16 if sd["image_proj"]["proj.weight"].shape[0] == 65536 else 4 for sd in state_dicts]

        with init_context():
            attn_procs[name] = FluxIPAdapterAttnProcessor(
                hidden_size=model.inner_dim,
                cross_attention_dim=model.config.joint_attention_dim,
                scale=1.0,
                num_tokens=num_image_text_embeds,
                dtype=model.dtype,
                device=model.device,
            )

        value_dict = {}
        for i, sd in enumerate(state_dicts):
            value_dict[f"to_k_ip.{i}.weight"] = sd["ip_adapter"][f"{key_id}.to_k_ip.weight"]
            value_dict[f"to_v_ip.{i}.weight"] = sd["ip_adapter"][f"{key_id}.to_v_ip.weight"]
            value_dict[f"to_k_ip.{i}.bias"] = sd["ip_adapter"][f"{key_id}.to_k_ip.bias"]
            value_dict[f"to_v_ip.{i}.bias"] = sd["ip_adapter"][f"{key_id}.to_v_ip.bias"]

        if low_cpu_mem_usage:
            load_model_dict_into_meta(attn_procs[name], value_dict, device_map={"": model.device}, dtype=model.dtype)
        else:
            attn_procs[name].load_state_dict(value_dict)

        key_id += 1

    empty_device_cache()
    return attn_procs


class FluxIPAdapterMixin:
    """Flux-specific IP-Adapter loader. Mixed into :class:`FluxTransformer2DModel`."""

    _supports_ip_adapter = True

    @classmethod
    def _metadata(cls):
        """Contribute the ``_supports_ip_adapter`` row to the metadata describe() table."""
        return {
            "_supports_ip_adapter": (
                True,
                "True",
                "Supports loading IP-Adapter weights (image-conditioning adapters).",
                f"{DOCS_BASE}/using-diffusers/ip_adapter",
            )
        }

    def _load_ip_adapter_weights(self, state_dicts, low_cpu_mem_usage=_LOW_CPU_MEM_USAGE_DEFAULT):
        """Install IP-Adapter weights on the Flux transformer.

        ``state_dicts`` is a single state dict (or a list, for multi-adapter loading); each dict must contain
        ``"image_proj"`` and ``"ip_adapter"`` sub-dicts.
        """
        if not isinstance(state_dicts, list):
            state_dicts = [state_dicts]

        self.encoder_hid_proj = None

        attn_procs = _convert_attn_processors(self, state_dicts, low_cpu_mem_usage=low_cpu_mem_usage)
        self.set_attn_processor(attn_procs)

        image_projection_layers = []
        for state_dict in state_dicts:
            image_projection_layer = _convert_image_proj(
                self, state_dict["image_proj"], low_cpu_mem_usage=low_cpu_mem_usage
            )
            image_projection_layers.append(image_projection_layer)

        self.encoder_hid_proj = MultiIPAdapterImageProjection(image_projection_layers)
        self.config.encoder_hid_dim_type = "ip_image_proj"
