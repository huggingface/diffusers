# Copyright 2026 Lightricks and The HuggingFace Team. All rights reserved.
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
import torch.nn as nn

from ...configuration_utils import ConfigMixin, register_to_config
from ...models.attention import AttentionModuleMixin
from ...models.attention_dispatch import dispatch_attention_fn
from ...models.modeling_utils import ModelMixin
from ...utils import logging


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class LTX2DurationAttnProcessor:
    """
    Processor for [`LTX2DurationAttentionPooler`].

    No attention mask is used: the text connectors substitute learnable registers for padded positions and mark the
    result fully attendable, so every token reaching this module is already valid.
    """

    _attention_backend = None
    _parallel_config = None

    def __call__(self, attn: "LTX2DurationAttentionPooler", tokens: torch.Tensor) -> torch.Tensor:
        queries = attn.query_tokens.unsqueeze(0).expand(tokens.shape[0], -1, -1)

        query = attn.to_q(queries).unflatten(2, (attn.heads, -1))
        key = attn.to_k(tokens).unflatten(2, (attn.heads, -1))
        value = attn.to_v(tokens).unflatten(2, (attn.heads, -1))

        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        hidden_states = hidden_states.flatten(2, 3)

        return attn.to_out(hidden_states)


class LTX2DurationAttentionPooler(nn.Module, AttentionModuleMixin):
    """
    Cross-attends `num_queries` learnable tokens against the caption tokens, producing a fixed `(batch_size,
    num_queries, hidden_dim)` output regardless of the input sequence length.
    """

    _default_processor_cls = LTX2DurationAttnProcessor
    _available_processors = [LTX2DurationAttnProcessor]

    def __init__(self, hidden_dim: int = 256, num_queries: int = 1, num_heads: int = 4):
        super().__init__()
        self.heads = num_heads
        self.query_tokens = nn.Parameter(torch.randn(num_queries, hidden_dim) * 0.02)
        self.to_q = nn.Linear(hidden_dim, hidden_dim)
        self.to_k = nn.Linear(hidden_dim, hidden_dim)
        self.to_v = nn.Linear(hidden_dim, hidden_dim)
        self.to_out = nn.Linear(hidden_dim, hidden_dim)
        self.set_processor(LTX2DurationAttnProcessor())

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.processor(self, tokens)


class LTX2DurationHead(ModelMixin, ConfigMixin):
    """
    Predicts the natural duration of the shot implied by a caption, from the LTX-2 text connector outputs.

    The head is modality-agnostic: pass either or both of the video and audio connector outputs. Modality-specific
    input projections map each stream into a shared pooler dimension, learnable modality embeddings tag the streams so
    the pooler can tell them apart, and a small MLP turns the pooled vector into a log-duration. The regression target
    is trained in log-seconds, so `forward` exponentiates and callers always get seconds.

    Ships from LTX-2.5 checkpoints onward.

    Args:
        video_cross_attention_dim (`int`, defaults to `4096`):
            Width of the video connector output.
        audio_cross_attention_dim (`int`, defaults to `2048`):
            Width of the audio connector output.
        pooler_hidden_dim (`int`, defaults to `256`):
            Shared hidden dimension both modalities are projected into.
        num_queries (`int`, defaults to `1`):
            Number of learnable pooling queries.
        num_pooler_heads (`int`, defaults to `4`):
            Attention heads used by the pooler.
        mlp_hidden_dim (`int`, defaults to `256`):
            Hidden width of the output MLP. Named with a `_dim` suffix to avoid colliding with the `mlp_hidden`
            submodule, which `ConfigMixin.__getattr__` would otherwise shadow with this config value.
    """

    @register_to_config
    def __init__(
        self,
        video_cross_attention_dim: int = 4096,
        audio_cross_attention_dim: int = 2048,
        pooler_hidden_dim: int = 256,
        num_queries: int = 1,
        num_pooler_heads: int = 4,
        mlp_hidden_dim: int = 256,
    ):
        super().__init__()

        self.video_input_proj = nn.Linear(video_cross_attention_dim, pooler_hidden_dim)
        self.video_modality_emb = nn.Parameter(torch.randn(pooler_hidden_dim) * 0.02)

        self.audio_input_proj = nn.Linear(audio_cross_attention_dim, pooler_hidden_dim)
        self.audio_modality_emb = nn.Parameter(torch.randn(pooler_hidden_dim) * 0.02)

        self.attention_pooler = LTX2DurationAttentionPooler(
            hidden_dim=pooler_hidden_dim,
            num_queries=num_queries,
            num_heads=num_pooler_heads,
        )
        self.mlp_hidden = nn.Linear(pooler_hidden_dim * num_queries, mlp_hidden_dim)
        self.mlp_out = nn.Linear(mlp_hidden_dim, 1)

    def forward(
        self,
        video_tokens: torch.Tensor | None = None,
        audio_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            video_tokens (`torch.Tensor` of shape `(batch_size, seq_len, video_cross_attention_dim)`, *optional*):
                Video connector output.
            audio_tokens (`torch.Tensor` of shape `(batch_size, seq_len, audio_cross_attention_dim)`, *optional*):
                Audio connector output.

        Returns:
            `torch.Tensor` of shape `(batch_size,)`: the predicted duration in seconds.
        """
        if video_tokens is None and audio_tokens is None:
            raise ValueError("`LTX2DurationHead` requires at least one of `video_tokens` / `audio_tokens`.")

        # Connector output can arrive in a different dtype than the head -- an fp32 text encoder feeding a bf16
        # head, say. Both reference implementations cast the inputs the same way. `self.dtype` rather than a
        # weight's dtype: a stored weight's dtype is not the compute dtype under quantized loading.
        head_dtype = self.dtype

        token_groups = []
        if video_tokens is not None:
            token_groups.append(self.video_input_proj(video_tokens.to(head_dtype)) + self.video_modality_emb)
        if audio_tokens is not None:
            token_groups.append(self.audio_input_proj(audio_tokens.to(head_dtype)) + self.audio_modality_emb)

        tokens = torch.cat(token_groups, dim=1)
        pooled = self.attention_pooler(tokens).flatten(1)

        # The tanh-approximated GELU matches the JAX-trained head; the exact GELU gives different numbers.
        hidden_states = torch.nn.functional.gelu(self.mlp_hidden(pooled), approximate="tanh")
        log_duration = self.mlp_out(hidden_states).squeeze(-1)

        return log_duration.exp()

    def predict_num_frames(
        self,
        video_tokens: torch.Tensor | None = None,
        audio_tokens: torch.Tensor | None = None,
        *,
        frame_rate: float,
        temporal_compression_ratio: int,
        min_seconds: float = 1.0,
        max_seconds: float = 20.0,
    ) -> int:
        """
        Predicts a frame count from connector tokens, clamped to `[min_seconds, max_seconds]` and snapped to the VAE's
        causal temporal grid (`k * temporal_compression_ratio + 1`).

        The clamp is applied before snapping: a clamped frame count is not necessarily grid-aligned, so snapping first
        would give a different result. Because snapping floors, it can land below the minimum; when that happens the
        result is snapped up to the next grid point instead, so the frame count stays within bounds.

        Narrow bounds can convert to a frame window containing no grid point at all -- at 24 fps, `[1.0s, 1.02s]`
        rounds to `[24, 24]`, and 24 is not `8k + 1`. The nearest grid point is used and a warning is logged, since
        overshooting by under one grid step beats refusing to generate. The returned count is therefore always on the
        grid, but may fall just outside the requested bounds in this case.

        Args:
            video_tokens (`torch.Tensor`, *optional*):
                Video connector output for a single prompt.
            audio_tokens (`torch.Tensor`, *optional*):
                Audio connector output for a single prompt.
            frame_rate (`float`):
                Frames per second used to convert the predicted duration into a frame count.
            temporal_compression_ratio (`int`):
                The VAE's temporal compression ratio, which defines the frame grid.
            min_seconds (`float`, defaults to `1.0`):
                Lower bound on the prediction.
            max_seconds (`float`, defaults to `20.0`):
                Upper bound on the prediction.

        Returns:
            `int`: a frame count lying on the VAE's temporal grid.
        """
        predicted_seconds = self(video_tokens, audio_tokens)
        if predicted_seconds.numel() != 1:
            raise ValueError(
                "`predict_num_frames` supports a single prediction only, but got a prediction of shape"
                f" {tuple(predicted_seconds.shape)}. One frame count cannot serve prompts with different natural"
                " durations -- predict for one prompt at a time."
            )
        seconds = predicted_seconds.item()

        # `min_frames` is floored at 1 so the grid arithmetic below cannot go negative, matching the reference's
        # `snap_frames_to_grid`, which rejects frame counts under 1.
        min_frames = max(1, round(min_seconds * frame_rate))
        max_frames = round(max_seconds * frame_rate)
        clamped_frames = max(min_frames, min(round(seconds * frame_rate), max_frames))

        num_frames = ((clamped_frames - 1) // temporal_compression_ratio) * temporal_compression_ratio + 1
        if num_frames < min_frames:
            # Flooring undershot the lower bound. The next grid point up is the in-bounds choice.
            snapped_up = num_frames + temporal_compression_ratio
            if snapped_up <= max_frames:
                num_frames = snapped_up
            else:
                # Converting the bounds to frames left a window with no grid point inside it at all, so they
                # cannot be honoured exactly. Take whichever neighbouring grid point is closest to the requested
                # length -- overshooting by under one grid step is a better answer than refusing to generate.
                if abs(snapped_up - clamped_frames) < abs(num_frames - clamped_frames):
                    num_frames = snapped_up
                logger.warning(
                    f"Duration bounds [{min_seconds:.2f}s, {max_seconds:.2f}s] at {frame_rate:.2f} fps admit no frame"
                    f" count on the VAE's temporal grid (k * {temporal_compression_ratio} + 1); using the nearest:"
                    f" {num_frames} frames ({num_frames / frame_rate:.2f}s)"
                )

        if seconds < min_seconds or seconds > max_seconds:
            logger.warning(
                f"Duration prediction clamped: raw {seconds:.2f}s outside [{min_seconds:.2f}s, {max_seconds:.2f}s],"
                f" using {num_frames / frame_rate:.2f}s ({num_frames} frames) @ {frame_rate:.2f} fps"
            )
        else:
            logger.info(f"Predicted duration {seconds:.2f}s ({num_frames} frames @ {frame_rate:.2f} fps)")

        return num_frames
