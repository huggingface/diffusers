import torch
import torch.nn as nn

from ...attention import BasicTransformerBlock
from ..transformer_2d import Transformer2DModel


class ContinuousTransformer2DModel(Transformer2DModel):
    def __init__(
        self,
        in_channels,
        out_channels,
        inner_dim,
        num_attention_heads,
        attention_head_dim,
        cross_attention_dim,
        activation_fn,
        num_embeds_ada_norm,
        attention_bias,
        only_cross_attention,
        double_self_attention,
        upcast_attention,
        norm_type,
        norm_elementwise_affine,
        norm_eps,
        attention_type,
        norm_num_groups,
        use_linear_projection,
        dropout,
        num_layers,
    ):
        self.in_channels = in_channels

        self.norm = torch.nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True)
        if use_linear_projection:
            self.proj_in = torch.nn.Linear(in_channels, inner_dim)
        else:
            self.proj_in = torch.nn.Conv2d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    num_embeds_ada_norm=num_embeds_ada_norm,
                    attention_bias=attention_bias,
                    only_cross_attention=only_cross_attention,
                    double_self_attention=double_self_attention,
                    upcast_attention=upcast_attention,
                    norm_type=norm_type,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                    attention_type=attention_type,
                )
                for d in range(num_layers)
            ]
        )

        self.out_channels = in_channels if out_channels is None else out_channels

        # TODO: should use out_channels for continuous projections
        if use_linear_projection:
            self.proj_out = torch.nn.Linear(inner_dim, in_channels)
        else:
            self.proj_out = torch.nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)
