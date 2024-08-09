from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from diffusers.models import UNetSpatioTemporalConditionModel
from diffusers.models.attention import TemporalBasicTransformerBlock, _chunked_feed_forward
from diffusers.utils.torch_utils import maybe_allow_in_graph


@maybe_allow_in_graph
def _forward_temporal_basic_transformer_block(
    self,
    camera_pose: torch.FloatTensor,
    scale: float,
    hidden_states: torch.FloatTensor,
    num_frames: int,
    encoder_hidden_states: Optional[torch.FloatTensor] = None,
) -> torch.FloatTensor:
    # Notice that normalization is always applied before the real computation in the following blocks.
    # 0. Self-Attention
    batch_size = hidden_states.shape[0]

    batch_frames, seq_length, channels = hidden_states.shape
    batch_size = batch_frames // num_frames

    hidden_states = hidden_states[None, :].reshape(batch_size, num_frames, seq_length, channels)
    hidden_states = hidden_states.permute(0, 2, 1, 3)
    hidden_states = hidden_states.reshape(batch_size * seq_length, num_frames, channels)

    residual = hidden_states
    hidden_states = self.norm_in(hidden_states)

    if self._chunk_size is not None:
        hidden_states = _chunked_feed_forward(self.ff_in, hidden_states, self._chunk_dim, self._chunk_size)
    else:
        hidden_states = self.ff_in(hidden_states)

    if self.is_res:
        hidden_states = hidden_states + residual

    norm_hidden_states = self.norm1(hidden_states)
    attn_output = self.attn1(norm_hidden_states, encoder_hidden_states=None)
    hidden_states = attn_output + hidden_states

    # MotionCtrl specific
    camera_pose = camera_pose.repeat_interleave(seq_length, dim=0)  # [batch_size * seq_length, num_frames, 12]
    residual = hidden_states
    hidden_states = torch.cat([hidden_states, camera_pose], dim=-1)
    hidden_states = scale * self.cc_projection(hidden_states) + (1 - scale) * residual

    # 3. Cross-Attention
    if self.attn2 is not None:
        norm_hidden_states = self.norm2(hidden_states)
        attn_output = self.attn2(norm_hidden_states, encoder_hidden_states=encoder_hidden_states)
        hidden_states = attn_output + hidden_states

    # 4. Feed-forward
    norm_hidden_states = self.norm3(hidden_states)

    if self._chunk_size is not None:
        ff_output = _chunked_feed_forward(self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size)
    else:
        ff_output = self.ff(norm_hidden_states)

    if self.is_res:
        hidden_states = ff_output + hidden_states
    else:
        hidden_states = ff_output

    hidden_states = hidden_states[None, :].reshape(batch_size, seq_length, num_frames, channels)
    hidden_states = hidden_states.permute(0, 2, 1, 3)
    hidden_states = hidden_states.reshape(batch_size * num_frames, seq_length, channels)

    return hidden_states


class UNetSpatioTemporalConditionMotionCtrlModel(UNetSpatioTemporalConditionModel):
    r"""UNetSpatioTemporalConditionModel for [MotionCtrl SVD](https://arxiv.org/abs/2312.03641)."""

    def __init__(self, motionctrl_kwargs: Dict[str, Any], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.motionctrl_scale = 1
        self._camera_pose = None

        camera_pose_embed_dim = motionctrl_kwargs.get("camera_pose_embed_dim")
        camera_pose_dim = motionctrl_kwargs.get("camera_pose_dim")

        def pre_hook(module, args):
            return (self._camera_pose, self.motionctrl_scale, *args)

        for _, module in self.named_modules():
            if isinstance(module, TemporalBasicTransformerBlock):
                cc_projection = nn.Linear(
                    module.time_mix_inner_dim + camera_pose_embed_dim * camera_pose_dim, module.time_mix_inner_dim
                )
                module.add_module("cc_projection", cc_projection)

                new_forward = _forward_temporal_basic_transformer_block.__get__(module, module.__class__)
                setattr(module, "forward", new_forward)
                module.register_forward_pre_hook(pre_hook)

    def set_motionctrl_scale(self, scale: float):
        self.motionctrl_scale = scale

    def forward(self, camera_pose: torch.FloatTensor, *args, **kwargs):
        self._camera_pose = camera_pose
        return super().forward(*args, **kwargs)
