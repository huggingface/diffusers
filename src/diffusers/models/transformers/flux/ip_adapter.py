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
"""Flux IP-Adapter conversion.

Per-model converters consumed by ``IPAdapterModelMixin`` via ``FLUX_IP_ADAPTER_METADATA``:

- ``convert_image_proj``: rewrites ``proj.weight`` → ``image_embeds.weight`` and builds an ``ImageProjection`` sized
  off the source state dict (4 or 16 image-text embeds depending on the ``proj.weight`` row count).
- ``convert_attn_processors``: walks ``model.attn_processors``, skips ``single_transformer_blocks`` (Flux only attaches
  IP-Adapter on the double-stream blocks), and builds one ``FluxIPAdapterAttnProcessor`` per remaining block. Reads
  ``model.config.joint_attention_dim`` and ``model.inner_dim`` for the projection dimensions and pulls ``to_k_ip`` /
  ``to_v_ip`` weights/biases keyed by ``key_id``.
"""

from contextlib import nullcontext

from ....loaders.ip_adapter_model import IPAdapterHandler
from ....models.embeddings import ImageProjection
from ....models.model_loading_utils import load_model_dict_into_meta
from ....models.modeling_utils import _LOW_CPU_MEM_USAGE_DEFAULT
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


def convert_image_proj(model, state_dict, low_cpu_mem_usage=_LOW_CPU_MEM_USAGE_DEFAULT):
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


def convert_attn_processors(model, state_dicts, low_cpu_mem_usage=_LOW_CPU_MEM_USAGE_DEFAULT):
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


# Handler assembled into ``ModelMetadata`` by ``flux/model.py``.
FLUX_IP_ADAPTER = IPAdapterHandler(
    convert_attn_to_diffusers_fn=convert_attn_processors,
    convert_image_proj_to_diffusers_fn=convert_image_proj,
)
