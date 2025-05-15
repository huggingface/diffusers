import torch
from torch import nn
from ..attention import BasicTransformerBlock, TemporalBasicTransformerBlock


class SkipConnect(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self, x_spatial: torch.Tensor, x_temporal: torch.Tensor
    ) -> torch.Tensor:
        return x_spatial + x_temporal


class SevaMultiviewTransformer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_heads: int,
        dim_head: int,
        name: str,
        unflatten_names: list[str] = [],
        transformer_depth: int = 1,
        context_dim: int = 1024,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.name = name
        self.unflatten_names = unflatten_names

        self.in_channels = in_channels
        inner_dim = num_heads * dim_head
        self.norm = nn.GroupNorm(32, in_channels, eps=1e-6)
        self.proj_in = nn.Linear(in_channels, inner_dim)
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    dim=inner_dim,
                    num_attention_heads=num_heads,
                    attention_head_dim=dim_head,
                    dropout=dropout,
                    cross_attention_dim=context_dim,
                )
            ]
        )
        self.proj_out = nn.Linear(inner_dim, in_channels)

        self.time_mixer = SkipConnect()
        time_mix_inner_dim = inner_dim
        self.time_mix_blocks = nn.ModuleList(
            [
                TemporalBasicTransformerBlock(
                    dim=inner_dim,
                    time_mix_inner_dim=time_mix_inner_dim,
                    num_attention_heads=num_heads,
                    attention_head_dim=dim_head,
                    dropout=dropout,
                    cross_attention_dim=context_dim,
                )
                for _ in range(transformer_depth)
            ]
        )

    def forward(
        self, x: torch.Tensor, context_emb: torch.Tensor, num_frames: int
    ) -> torch.Tensor:
        assert context_emb.ndim == 3
        _, _, height, width = x.shape

        time_context = context_emb
        time_context_first_timestep = time_context[::num_frames]
        time_context = torch.repeat_interleave(
            time_context_first_timestep, height * width, dim=0
        )

        h = self.norm(x)
        h = h.permute(0, 2, 3, 1).contiguous()
        h = h.view(h.shape[0], -1, h.shape[3])
        h = self.proj_in(h)

        for transformer_block, time_mix_block in zip(self.transformer_blocks, self.time_mix_blocks):
            if self.name in self.unflatten_names:
                context_emb = context_emb[::num_frames]
                context_emb = context_emb.view(
                    context_emb.shape[0]//num_frames, context_emb[1]*num_frames, context_emb.shape[2]
                )
            
            h = transformer_block(h, encoder_hidden_states=context_emb)

            if self.name in self.unflatten_names:
                context_emb = context_emb.view(
                    context_emb.shape[0]*num_frames, context_emb[1]//num_frames, context_emb.shape[2]
                )

            h_mix = time_mix_block(h, encoder_hidden_states=time_context, num_frames=num_frames)    
            h = self.time_mixer(x_spatial=h, x_temporal=h_mix)
    
        h = self.proj_out(h)
        h = h.view(h.shape[0], height, width, h.shape[2])
        h = h.permute(0, 3, 1, 2).contiguous()
        out = h + x
        return out
