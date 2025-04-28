import torch
from torch import nn
from typing import Optional
from dataclasses import field
from torch.nn import functional as F
from ..modeling_utils import ModelMixin
from ...configuration_utils import ConfigMixin, register_to_config
from ..transformers import SevaMultiviewTransformer
from ..embeddings import TimestepEmbedding, Timesteps


class SevaUpSampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: Optional[int]):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.conv = nn.Conv2d(self.in_channels, self.out_channels, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1] == self.in_channels
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class SevaDownSampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: Optional[int]):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.conv = nn.Conv2d(self.in_channels, self.out_channels, 3, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1] == self.in_channels
        return self.conv(x)


class GroupNorm32(nn.GroupNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).type(x.dtype)


class SevaResNetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        emb_channels: int,
        out_channels: Optional[int],
        dense_in_channels: int,
        dropout: float,
    ):
        super().__init__()
        out_channels = out_channels or in_channels

        self.in_layers = nn.Sequential(
            GroupNorm32(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
        )
        self.emb_layers = nn.Sequential(
            nn.SiLU(), nn.Linear(emb_channels, out_channels)
        )
        self.dense_emb_layers = nn.Sequential(
            nn.Conv2d(dense_in_channels, 2 * in_channels, 1, 1, 0)
        )
        self.out_layers = nn.Sequential(
            GroupNorm32(32, out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
        )
        if out_channels == in_channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

    def forward(
        self, x: torch.Tensor, time_emb: torch.Tensor, dense_emb: torch.Tensor
    ) -> torch.Tensor:
        input_dtype = x.dtype
        in_activated_norm, in_conv = self.in_layers[:-1], self.in_layers[-1]
        dense_transformed = self.dense_emb_layers(
            F.interpolate(
                dense_emb, size=x.shape[2:], mode="bilinear", align_corners=True
            )
        ).to(input_dtype)
        dense_scale, dense_shift = torch.chunk(dense_transformed, 2, dim=1)
        time_emb_out = self.emb_layers(time_emb).to(input_dtype)

        h = in_activated_norm(x)
        h = h * (1 + dense_scale) + dense_shift
        h = in_conv(h)
        while len(time_emb_out.shape) < len(h.shape):
            time_emb_out = time_emb_out[..., None]
        h = h + time_emb_out
        h = self.out_layers(h)
        h = self.skip_connection(x) + h
        return h


class SevaBlock(nn.Sequential):
    def forward(
        self,
        x: torch.Tensor,
        time_emb: torch.Tensor,
        context_emb: torch.Tensor,
        dense_emb: torch.Tensor,
        num_frames: int,
    ) -> torch.Tensor:
        for layer in self:
            if isinstance(layer, SevaMultiviewTransformer):
                assert num_frames is not None
                x = layer(x, context_emb, num_frames)
            elif isinstance(layer, SevaResNetBlock):
                x = layer(x, time_emb, dense_emb)
            else:
                x = layer(x)
        return x


class SevaUnet(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        in_channels: int = 11,
        model_channels: int = 320,
        out_channels: int = 4,
        num_frames: int = 21,
        num_resnet_blocks: int = 2,
        attention_resolutions: list[int] = field(default_factory=lambda: [4, 2, 1]),
        channel_multipliers: list[int] = field(default_factory=lambda: [1, 2, 4, 4]),
        channels_per_head: int = 64,
        transformer_depth: list[int] = field(default_factory=lambda: [1, 1, 1, 1]),
        context_dim: int = 1024,
        dense_in_channels: int = 6,
        dropout: float = 0.0,
        unflatten_names: list[str] = field(
            default_factory=lambda: ["middle_ds8", "output_ds4", "output_ds2"]
        ),
    ) -> None:
        super().__init__()

        # Time Embedding Block
        self.time_proj = Timesteps(
            model_channels, flip_sin_to_cos=True, downscale_freq_shift=0
        )
        time_embedding_dim = model_channels * 4
        self.time_embedding = TimestepEmbedding(
            model_channels,
            time_embedding_dim,
        )

        # Input Block
        current_channels = model_channels
        input_block_chans = [current_channels]
        self.input_blocks = nn.ModuleList(
            [SevaBlock(nn.Conv2d(in_channels, current_channels, 3, padding=1))]
        )
        for level, channel_mult in enumerate(channel_multipliers):
            for _ in range(num_resnet_blocks):
                input_layers = [
                    SevaResNetBlock(
                        channels=current_channels,
                        emb_channels=time_embedding_dim,
                        out_channels=model_channels * channel_mult,
                        dense_in_channels=dense_in_channels,
                        dropout=dropout,
                    )
                ]
                current_channels = current_channels * channel_mult
                ds = 2**level
                if ds in attention_resolutions:
                    num_heads = current_channels // channels_per_head
                    dim_head = channels_per_head
                    input_layers.append(
                        SevaMultiviewTransformer(
                            current_channels,
                            num_heads,
                            dim_head,
                            name=f"input_ds{ds}",
                            depth=transformer_depth[level],
                            context_dim=context_dim,
                            unflatten_names=unflatten_names,
                        )
                    )
                self.input_blocks.append(SevaBlock(*input_layers))
                input_block_chans.append(current_channels)

            if level != len(channel_multipliers) - 1:
                self.input_blocks.append(
                    SevaBlock(
                        SevaDownSampleBlock(
                            current_channels, out_channels=current_channels
                        )
                    )
                )
                input_block_chans.append(current_channels)

        # Middle Block
        num_heads = current_channels // channels_per_head
        dim_head = channels_per_head
        ds = 2 ** (len(channel_multipliers))
        self.middle_block = SevaBlock(
            SevaResNetBlock(
                channels=current_channels,
                emb_channels=time_embedding_dim,
                out_channels=None,
                dense_in_channels=dense_in_channels,
                dropout=dropout,
            ),
            SevaMultiviewTransformer(
                current_channels,
                num_heads,
                dim_head,
                name=f"middle_ds{ds}",
                depth=transformer_depth[-1],
                context_dim=context_dim,
                unflatten_names=unflatten_names,
            ),
            SevaResNetBlock(
                channels=current_channels,
                emb_channels=time_embedding_dim,
                out_channels=None,
                dense_in_channels=dense_in_channels,
                dropout=dropout,
            ),
        )

        # Output Block
        self.output_blocks = nn.ModuleList([])
        for level, channel_mult in list(enumerate(channel_multipliers))[::-1]:
            for idx in range(num_resnet_blocks + 1):
                input_block_channel = input_block_chans.pop()
                current_channels = current_channels + input_block_channel
                output_layers = [
                    SevaResNetBlock(
                        channels=current_channels,
                        emb_channels=time_embedding_dim,
                        out_channels=model_channels * channel_mult,
                        dense_in_channels=dense_in_channels,
                        dropout=dropout,
                    )
                ]

                current_channels = model_channels * channel_mult
                if ds in attention_resolutions:
                    num_heads = current_channels // channels_per_head
                    dim_head = channels_per_head
                    output_layers.append(
                        SevaMultiviewTransformer(
                            current_channels,
                            num_heads,
                            dim_head,
                            name=f"output_ds{ds}",
                            depth=transformer_depth[level],
                            context_dim=context_dim,
                            unflatten_names=unflatten_names,
                        )
                    )
                if level and idx == num_resnet_blocks:
                    ds //= 2
                    output_layers.append(
                        SevaUpSampleBlock(
                            current_channels, out_channels=current_channels
                        )
                    )
                self.output_blocks.append(SevaBlock(*output_layers))

        # Final Output Block
        self.out = nn.Sequential(
            GroupNorm32(32, current_channels),
            nn.SiLU(),
            nn.Conv2d(model_channels, out_channels, 3, padding=1),
        )

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context_emb: torch.Tensor,
        dense_emb: torch.Tensor,
        num_frames: Optional[int] = None,
    ) -> torch.Tensor:
        num_frames = num_frames or self.config.num_frames
        time_emb = self.time_proj(timestep)
        time_emb = self.time_embedding(time_emb)

        input_dtype = x.dtype
        h = x
        input_block_embs = []
        for module in self.input_blocks:
            h = module(
                h,
                time_emb=time_emb,
                context_emb=context_emb,
                dense_emb=dense_emb,
                num_frames=num_frames,
            )
            input_block_embs.append(h)

        h = self.middle_block(
            h,
            time_emb=time_emb,
            context_emb=context_emb,
            dense_emb=dense_emb,
            num_frames=num_frames,
        )

        for module in self.output_blocks:
            h = torch.cat([h, input_block_embs.pop()], dim=1)
            h = module(
                h,
                time_emb=time_emb,
                context_emb=context_emb,
                dense_emb=dense_emb,
                num_frames=num_frames,
            )

        h = h.to(input_dtype)
        return self.out(h)
