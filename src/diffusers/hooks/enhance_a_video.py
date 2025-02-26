# Copyright 2024 The HuggingFace Team. All rights reserved.
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

import re
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, Union

import torch
import torch.overrides

from ..utils import get_logger
from ._common import _ATTENTION_CLASSES
from .hooks import HookRegistry, ModelHook


logger = get_logger(__name__)


_ENHANCE_A_VIDEO = "enhance_a_video"


class _AttentionType(int, Enum):
    SELF = 0
    JOINT___LATENTS_FIRST = 1
    JOINT___LATENTS_LAST = 2


@dataclass
class EnhanceAVideoConfig:
    r"""
    Configuration for [Enhance A Video](https://huggingface.co/papers/2502.07508).

    Args:
        weight (`float` or `Dict[str, float]`, defaults to `1.0`):
            The weighting factor for the Enhance A Video score. If a `float`, the same weight is applied to all layers.
            If a `dict`, the keys are regex patterns that match non-overlapping layer names, and the values are the
            corresponding weights.
        num_frames_callback (`Callable[[], int]`, `optional`):
            A callback function that returns the number of latent frames in the latent video stream. Since there is no
            easy way to deduce this within the attention layers, the user must provide this information.
        _attention_type (`_AttentionType`, defaults to `_AttentionType.SELF`):
            The type of attention mechanism that the underlying model uses. The following options are available:
                - `_AttentionType.SELF`:
                    The model uses self-attention layers with only video tokens.
                - `_AttentionType.JOINT___LATENTS_FIRST`:
                    The model uses joint attention layers (concatenated video and text stream data) with video tokens
                    first.
                - `_AttentionType.JOINT___LATENTS_LAST`:
                    The model uses joint attention layers (concatenated video and text stream data) with video tokens
                    last.

            This parameter is not backwards-compatible and may be subject to change in future versions.
    """

    weight: Union[float, Dict[str, float]] = 1.0
    num_frames_callback: Callable[[], int] = None
    _attention_type: _AttentionType = _AttentionType.SELF


class EnhanceAVideoAttentionState:
    def __init__(self) -> None:
        self.query: torch.Tensor = None
        self.key: torch.Tensor = None
        self.latents_sequence_length: int = None

    def reset(self) -> None:
        self.query = None
        self.key = None
        self.latents_sequence_length = None

    def __repr__(self):
        return f"EnhanceAVideoAttentionState(latents_sequence_length={self.latents_sequence_length})"


class EnhanceAVideoCaptureSDPAInputsFunctionMode(torch.overrides.TorchFunctionMode):
    def __init__(self, query_key_save_callback: Callable[[torch.Tensor, torch.Tensor], None]) -> None:
        super().__init__()

        self.query_key_save_callback = query_key_save_callback

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        # TODO(aryan): revisit for torch.compile -- can trace into ATen but not triton, so this branch is never hit
        if func is torch.nn.functional.scaled_dot_product_attention:
            query = kwargs.get("query", None) or args[0]
            key = kwargs.get("key", None) or args[1]
            self.query_key_save_callback(query, key)
        return func(*args, **kwargs)


class EnhanceAVideoSDPAHook(ModelHook):
    _is_stateful = True

    def __init__(self, weight: float, num_frames_callback: Callable[[], int], _attention_type: _AttentionType) -> None:
        self.weight = weight
        self.num_frames_callback = num_frames_callback
        self._attention_type = _attention_type

    def initialize_hook(self, module):
        self.state = EnhanceAVideoAttentionState()
        return module

    def new_forward(self, module, *args, **kwargs):
        # Here, query and key have two shapes (considering the general diffusers-style model implementation):
        #   1. [batch_size, attention_heads, latents_sequence_length, head_dim]
        #   2. [batch_size, attention_heads, latents_sequence_length + encoder_sequence_length, head_dim]
        #   3. [batch_size, attention_heads, encoder_sequence_length + latents_sequence_length, head_dim]
        kwargs_hidden_states = kwargs.get("hidden_states", None)
        hidden_states = kwargs_hidden_states if kwargs_hidden_states is not None else args[0]
        self.state.latents_sequence_length = hidden_states.size(1)

        # Capture query and key tensors to compute EnhanceAVideo scores
        with EnhanceAVideoCaptureSDPAInputsFunctionMode(self._query_key_capture_callback):
            return self.fn_ref.original_forward(*args, **kwargs)

    def post_forward(self, module, output):
        # For diffusers models, or ones that are implemented similar to our design, we either return:
        #   1. A single output: `hidden_states`
        #   2. A tuple of outputs: `(hidden_states, encoder_hidden_states)`.
        # We need to handle both cases of applying EnhanceAVideo scores.
        hidden_states = output[0] if isinstance(output, tuple) else output

        def reshape_for_framewise_attention(tensor: torch.Tensor) -> torch.Tensor:
            # This code assumes tensor is [B, N, S, C]. This should be true for most diffusers-style implementations.
            # [B, N, S, C] -> [B, N, F, S, C] -> [B, S, N, F, C] -> [B * S, N, F, C]
            return tensor.unflatten(2, (num_frames, -1)).permute(0, 3, 1, 2, 4).flatten(0, 1)

        # Handle reshaping of query and key tensors
        query, key = self.state.query, self.state.key
        if self._attention_type == _AttentionType.SELF:
            pass
        elif self._attention_type == _AttentionType.JOINT___LATENTS_FIRST:
            query = query[:, :, : self.state.latents_sequence_length]
            key = key[:, :, : self.state.latents_sequence_length]
        elif self._attention_type == _AttentionType.JOINT___LATENTS_LAST:
            query = query[:, :, -self.state.latents_sequence_length :]
            key = key[:, :, -self.state.latents_sequence_length :]

        num_frames = self.num_frames_callback()
        query = reshape_for_framewise_attention(query)
        key = reshape_for_framewise_attention(key)
        scores = enhance_a_video_score(query, key, num_frames, self.weight)
        hidden_states = hidden_states * scores

        return (hidden_states, *output[1:]) if isinstance(output, tuple) else hidden_states

    def reset_state(self, module):
        self.state.reset()
        return module

    def _query_key_capture_callback(self, query: torch.Tensor, key: torch.Tensor) -> None:
        self.state.query = query
        self.state.key = key


def enhance_a_video_score(
    query: torch.Tensor, key: torch.Tensor, num_frames: int, weight: float = 1.0
) -> torch.Tensor:
    head_dim = query.size(-1)
    scale = 1 / (head_dim**0.5)
    query = query * scale

    attn_temp = query @ key.transpose(-2, -1)
    attn_temp = attn_temp.float()
    attn_temp = attn_temp.softmax(dim=-1)

    # Reshape to [batch_size * num_tokens, num_frames, num_frames]
    attn_temp = attn_temp.reshape(-1, num_frames, num_frames)

    # Create a mask for diagonal elements
    diag_mask = torch.eye(num_frames, device=attn_temp.device).bool()
    diag_mask = diag_mask.unsqueeze(0).expand(attn_temp.size(0), -1, -1)

    # Zero out diagonal elements
    attn_wo_diag = attn_temp.masked_fill(diag_mask, 0)

    # Calculate mean for each token's attention matrix
    # Number of off-diagonal elements per matrix is n*n - n
    num_off_diag = num_frames * num_frames - num_frames
    mean_scores = attn_wo_diag.sum(dim=(1, 2)) / num_off_diag

    scores = mean_scores.mean() * (num_frames + weight)
    scores = scores.clamp(min=1)
    return scores


def apply_enhance_a_video(module: torch.nn.Module, config: EnhanceAVideoConfig) -> None:
    r"""
    Applies [Enhance A Video](https://huggingface.co/papers/2502.07508) on a model.

    This function applies a Diffusers Hook to all/user-configured self-attention lyaers of the model. The hook captures
    the inputs entering `torch.nn.functional.scaled_dot_product_attention` and operates on them.

    Args:
        module (`torch.nn.Module`):
            The model to apply Enhance A Video to. It must be a video generation model.
        config (`EnhanceAVideoConfig`):
            The configuration for Enhance A Video.

    Example:

    ```python
    >>> import torch
    >>> from diffusers import LTXPipeline
    >>> from diffusers.hooks import apply_enhance_a_video, remove_enhance_a_video, EnhanceAVideoConfig
    >>> from diffusers.utils import export_to_video

    >>> pipe = LTXPipeline.from_pretrained("Lightricks/LTX-Video", torch_dtype=torch.bfloat16)
    >>> pipe.to("cuda")

    >>> weight = 4
    >>> num_frames = 161
    >>> latent_num_frames = (num_frames - 1) // pipe.vae_temporal_compression_ratio + 1

    >>> # Apply Enhance-A-Video to all layers with a weight of 4
    >>> config = EnhanceAVideoConfig(weight=weight, num_frames_callback=lambda: latent_num_frames, _attention_type=1)
    >>> apply_enhance_a_video(pipe.transformer, config)

    >>> prompt = "A man standing in a sunlit garden, surrounded by lush greenery and colorful flowers. The man has a knife in his hand and is cutting a ripe, juicy watermelon. The watermelon is bright red and contrasts beautifully with the green foliage in the background."
    >>> negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"

    >>> video = pipe(prompt=prompt, negative_prompt=negative_prompt, num_frames=161).frames[0]
    >>> export_to_video(video, "output.mp4", fps=24)

    >>> # Remove Enhance-A-Video
    >>> remove_enhance_a_video(pipe.transformer)

    >>> # Apply Enhance-A-Video to specific layers with different weights
    >>> config = EnhanceAVideoConfig(
    ...     weight={
    ...         "blocks\.(0|1|2|3|4|5|6|7)\.": 5.0,
    ...         "blocks\.(10|11|12|13|14|15)\.": 8.0,
    ...         "blocks\.(21|22|23|24|25|26)\.": 3.0,
    ...     },
    ...     num_frames_callback=lambda: latent_num_frames,
    ...     _attention_type=1,
    ... )
    ```
    """
    weight = config.weight
    if not isinstance(weight, dict):
        weight = {".*": config.weight}
    _validate_weight(module, weight)

    weight_keys = set(weight.keys())
    for name, submodule in module.named_modules():
        # We cannot apply Enhance-A-Video to cross-attention layers
        is_cross_attention = getattr(submodule, "is_cross_attention", False)
        if not isinstance(submodule, _ATTENTION_CLASSES) or is_cross_attention:
            continue
        current_weight = next(
            (weight[identifier] for identifier in weight_keys if re.search(identifier, name) is not None), None
        )
        if current_weight is None:
            continue
        logger.debug(f"Applying Enhance-A-Video to layer '{name}'")
        hook_registry = HookRegistry.check_if_exists_or_initialize(submodule)
        hook = EnhanceAVideoSDPAHook(
            weight=current_weight,
            num_frames_callback=config.num_frames_callback,
            _attention_type=config._attention_type,
        )
        hook_registry.register_hook(hook, _ENHANCE_A_VIDEO)


def remove_enhance_a_video(module: torch.nn.Module) -> None:
    r"""
    Removes the Enhance A Video hook from the model.

    See [`~hooks.enhance_a_video.apply_enhance_a_video`] for an example.
    """
    for name, submodule in module.named_modules():
        if not hasattr(submodule, "_diffusers_hook"):
            continue
        hook_registry = submodule._diffusers_hook
        hook_registry.remove_hook(_ENHANCE_A_VIDEO, recurse=False)
        logger.debug(f"Removed Enhance-A-Video from layer '{name}'")


def _validate_weight(module: torch.nn.Module, weight: Dict[str, float]) -> None:
    if not isinstance(weight, dict):
        raise ValueError(f"Invalid weight type: {type(weight)}")
    weight_keys = set(weight.keys())
    for name, _ in module.named_modules():
        num_matches = sum(re.search(identifier, name) is not None for identifier in weight_keys)
        if num_matches > 1:
            raise ValueError(
                f"The provided weight dictionary has multiple regex matches for layer '{name}'. Please provide non-overlapping regex patterns."
            )
