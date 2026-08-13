# Copyright 2026 Echo Team and The HuggingFace Team. All rights reserved.
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

"""Echo-Memory community pipeline for official Wan 2.1 Diffusers weights.

Loads `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`, then overlays the released
`context_k1` row from `Echo-Team/Echo-Memory` after remapping original
DiffSynth / Wan keys onto the Diffusers transformer.

Paper: https://arxiv.org/abs/2606.09803
Code: https://github.com/Echo-Team-Joy-Future-Academy-JD/Echo-Memory
"""

from typing import Dict, Iterable, List, Optional, Tuple

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from diffusers import WanPipeline


DEFAULT_BASE_MODEL = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
DEFAULT_REPO_ID = "Echo-Team/Echo-Memory"
DEFAULT_FILENAME = "context_k1/epoch-0.safetensors"
DEFAULT_CONVERTED_REPO_ID = "Wayne-King/echo-memory-diffusers"
DEFAULT_CONVERTED_FILENAME = "context_k1-diffusers/diffusion_pytorch_model.safetensors"

SKIP_SUBSTRINGS = (
    "action_mlp",
    "self_attn_with_action",
    "block_wise_ssm",
    "videossm_hybrid",
    "spatial_memory_module",
)

# Same mapping as `scripts/convert_wan_to_diffusers.py` for Wan 2.1 T2V.
TRANSFORMER_KEYS_RENAME_DICT = {
    "time_embedding.0": "condition_embedder.time_embedder.linear_1",
    "time_embedding.2": "condition_embedder.time_embedder.linear_2",
    "text_embedding.0": "condition_embedder.text_embedder.linear_1",
    "text_embedding.2": "condition_embedder.text_embedder.linear_2",
    "time_projection.1": "condition_embedder.time_proj",
    "head.modulation": "scale_shift_table",
    "head.head": "proj_out",
    "modulation": "scale_shift_table",
    "ffn.0": "ffn.net.0.proj",
    "ffn.2": "ffn.net.2",
    # The original model names norms as norm1, norm3, norm2.
    # Diffusers uses norm1, norm2, norm3.
    "norm2": "norm__placeholder",
    "norm3": "norm2",
    "norm__placeholder": "norm3",
    "self_attn.q": "attn1.to_q",
    "self_attn.k": "attn1.to_k",
    "self_attn.v": "attn1.to_v",
    "self_attn.o": "attn1.to_out.0",
    "self_attn.norm_q": "attn1.norm_q",
    "self_attn.norm_k": "attn1.norm_k",
    "cross_attn.q": "attn2.to_q",
    "cross_attn.k": "attn2.to_k",
    "cross_attn.v": "attn2.to_v",
    "cross_attn.o": "attn2.to_out.0",
    "cross_attn.norm_q": "attn2.norm_q",
    "cross_attn.norm_k": "attn2.norm_k",
}


def is_diffusers_transformer_state_dict(keys: Iterable[str]) -> bool:
    keys = list(keys)
    return any(key.startswith("condition_embedder.") or ".attn1." in key for key in keys)


def convert_echo_memory_transformer_state_dict(
    state_dict: Dict[str, torch.Tensor],
    skip_substrings: Iterable[str] = SKIP_SUBSTRINGS,
) -> Tuple[Dict[str, torch.Tensor], List[str]]:
    """Convert original Echo-Memory / DiffSynth Wan keys to Diffusers names."""
    skip_substrings = tuple(skip_substrings)
    if is_diffusers_transformer_state_dict(state_dict):
        converted = {
            key: value
            for key, value in state_dict.items()
            if not any(token in key for token in skip_substrings)
        }
        skipped = [key for key in state_dict if key not in converted]
        return converted, skipped

    converted = {}
    skipped = []
    for key, value in state_dict.items():
        if any(token in key for token in skip_substrings):
            skipped.append(key)
            continue
        new_key = key
        for replace_key, rename_key in TRANSFORMER_KEYS_RENAME_DICT.items():
            new_key = new_key.replace(replace_key, rename_key)
        converted[new_key] = value
    return converted, skipped


class EchoMemoryPipeline(WanPipeline):
    """Wan 2.1 T2V pipeline with an Echo-Memory `context_k1` overlay."""

    def load_echo_memory_weights(
        self,
        repo_id: str = DEFAULT_REPO_ID,
        filename: str = DEFAULT_FILENAME,
        local_path: Optional[str] = None,
        strict: bool = False,
    ):
        """Download one Echo-Memory row and overlay it on `self.transformer`."""
        ckpt_path = local_path or hf_hub_download(repo_id=repo_id, filename=filename)
        raw = load_file(ckpt_path)
        converted, skipped = convert_echo_memory_transformer_state_dict(raw)
        missing, unexpected = self.transformer.load_state_dict(converted, strict=strict)
        print(
            f"[Echo-Memory] overlaid {len(converted)}/{len(raw)} transformer keys from {ckpt_path} "
            f"(skipped={len(skipped)}, missing={len(missing)}, unexpected={len(unexpected)})"
        )
        return missing, unexpected, skipped

    def load_converted_echo_memory_weights(
        self,
        repo_id: str = DEFAULT_CONVERTED_REPO_ID,
        filename: str = DEFAULT_CONVERTED_FILENAME,
        local_path: Optional[str] = None,
        strict: bool = False,
    ):
        """Overlay the already-remapped `context_k1` transformer weights."""
        return self.load_echo_memory_weights(
            repo_id=repo_id,
            filename=filename,
            local_path=local_path,
            strict=strict,
        )

    @classmethod
    def from_echo_memory(
        cls,
        pretrained_model_name_or_path: str = DEFAULT_BASE_MODEL,
        echo_memory_repo: str = DEFAULT_REPO_ID,
        echo_memory_filename: str = DEFAULT_FILENAME,
        **kwargs,
    ):
        pipe = cls.from_pretrained(pretrained_model_name_or_path, **kwargs)
        pipe.load_echo_memory_weights(repo_id=echo_memory_repo, filename=echo_memory_filename)
        return pipe
