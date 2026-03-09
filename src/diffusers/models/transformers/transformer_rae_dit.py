from __future__ import annotations

from math import pi, sqrt

import torch
import torch.nn.functional as F
from torch import nn

from ...configuration_utils import ConfigMixin, register_to_config
from ..embeddings import PatchEmbed, get_2d_sincos_pos_embed
from ..modeling_outputs import Transformer2DModelOutput
from ..modeling_utils import ModelMixin
from ..normalization import RMSNorm


def _repeat_to_length(hidden_states: torch.Tensor, target_length: int) -> torch.Tensor:
    if hidden_states.shape[1] == target_length:
        return hidden_states

    if target_length % hidden_states.shape[1] != 0:
        raise ValueError(
            f"Cannot repeat sequence of length {hidden_states.shape[1]} to match target length {target_length}."
        )

    if hidden_states.shape[1] == 1:
        return hidden_states.expand(-1, target_length, -1)

    source_side = int(sqrt(hidden_states.shape[1]))
    target_side = int(sqrt(target_length))
    if (
        source_side * source_side == hidden_states.shape[1]
        and target_side * target_side == target_length
        and target_side % source_side == 0
    ):
        scale = target_side // source_side
        batch_size, _, channels = hidden_states.shape
        hidden_states = hidden_states.reshape(batch_size, source_side, source_side, channels)
        hidden_states = hidden_states.repeat_interleave(scale, dim=1).repeat_interleave(scale, dim=2)
        return hidden_states.reshape(batch_size, target_length, channels)

    raise ValueError(
        "Cannot expand conditioning tokens without preserving their 2D layout: "
        f"source length {hidden_states.shape[1]} is incompatible with target length {target_length}."
    )


def _ddt_modulate(hidden_states: torch.Tensor, shift: torch.Tensor | None, scale: torch.Tensor) -> torch.Tensor:
    if shift is None:
        shift = torch.zeros_like(scale)

    shift = _repeat_to_length(shift, hidden_states.shape[1])
    scale = _repeat_to_length(scale, hidden_states.shape[1])
    return hidden_states * (1 + scale) + shift


def _ddt_gate(hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    gate = _repeat_to_length(gate, hidden_states.shape[1])
    return hidden_states * gate


def _to_pair(value: int | tuple[int, int] | list[int], name: str) -> tuple[int, int]:
    if isinstance(value, int):
        return value, value

    if len(value) != 2:
        raise ValueError(f"`{name}` must be an int or a pair, but got {value}.")

    return int(value[0]), int(value[1])


def _rotate_half(hidden_states: torch.Tensor) -> torch.Tensor:
    hidden_states = hidden_states.view(*hidden_states.shape[:-1], -1, 2)
    first, second = hidden_states.unbind(dim=-1)
    hidden_states = torch.stack((-second, first), dim=-1)
    return hidden_states.flatten(-2)


class _ApproximateGELUMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(intermediate_size, hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class SwiGLUFFN(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.w12 = nn.Linear(hidden_size, 2 * intermediate_size)
        self.w3 = nn.Linear(intermediate_size, hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.w12(hidden_states)
        hidden_states_1, hidden_states_2 = hidden_states.chunk(2, dim=-1)
        hidden_states = F.silu(hidden_states_1) * hidden_states_2
        hidden_states = self.w3(hidden_states)
        return hidden_states


class GaussianFourierEmbedding(nn.Module):
    def __init__(self, hidden_size: int, embedding_size: int = 256, scale: float = 1.0):
        super().__init__()
        self.embedding_size = embedding_size
        self.scale = scale
        self.W = nn.Parameter(torch.randn(embedding_size) * scale, requires_grad=False)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_size * 2, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        timestep = timestep.to(self.W.dtype)
        hidden_states = timestep[:, None] * self.W[None, :] * 2 * pi
        hidden_states = torch.cat([torch.sin(hidden_states), torch.cos(hidden_states)], dim=-1)
        hidden_states = self.mlp(hidden_states)
        return hidden_states


class LabelEmbedder(nn.Module):
    def __init__(self, num_classes: int, hidden_size: int, dropout_prob: float):
        super().__init__()
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob
        self.embedding_table = nn.Embedding(num_classes + int(dropout_prob > 0), hidden_size)

    def token_drop(
        self, class_labels: torch.LongTensor, force_drop_ids: torch.Tensor | None = None
    ) -> torch.LongTensor:
        if force_drop_ids is None:
            drop_ids = torch.rand(class_labels.shape[0], device=class_labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        return torch.where(drop_ids, self.num_classes, class_labels)

    def forward(
        self,
        class_labels: torch.LongTensor,
        train: bool,
        force_drop_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (train and self.dropout_prob > 0) or (force_drop_ids is not None):
            class_labels = self.token_drop(class_labels, force_drop_ids=force_drop_ids)
        return self.embedding_table(class_labels)


class VisionRotaryEmbeddingFast(nn.Module):
    def __init__(self, dim: int, pt_seq_len: int, ft_seq_len: int | None = None, theta: float = 10000.0):
        super().__init__()

        if ft_seq_len is None:
            ft_seq_len = pt_seq_len

        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32)[: dim // 2] / dim))
        positions = torch.arange(ft_seq_len, dtype=torch.float32) / ft_seq_len * pt_seq_len
        freqs = torch.einsum("n,d->nd", positions, freqs)
        freqs = freqs.repeat_interleave(2, dim=-1)
        freqs = torch.cat(
            [
                freqs[:, None, :].expand(ft_seq_len, ft_seq_len, -1),
                freqs[None, :, :].expand(ft_seq_len, ft_seq_len, -1),
            ],
            dim=-1,
        )

        self.register_buffer("freqs_cos", freqs.cos().view(-1, freqs.shape[-1]))
        self.register_buffer("freqs_sin", freqs.sin().view(-1, freqs.shape[-1]))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        _, _, sequence_length, _ = hidden_states.shape
        base_sequence_length = self.freqs_cos.shape[0]
        repeat = sequence_length // base_sequence_length

        freqs_cos = self.freqs_cos
        freqs_sin = self.freqs_sin
        if repeat != 1:
            freqs_cos = freqs_cos.repeat_interleave(repeat, dim=0)
            freqs_sin = freqs_sin.repeat_interleave(repeat, dim=0)

        return hidden_states * freqs_cos + _rotate_half(hidden_states) * freqs_sin


class NormAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        use_rmsnorm: bool = False,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")

        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        norm_cls = RMSNorm if use_rmsnorm else nn.LayerNorm
        norm_kwargs = {"eps": 1e-6} if use_rmsnorm else {"elementwise_affine": True, "eps": 1e-6}

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_cls(self.head_dim, **norm_kwargs) if qk_norm else nn.Identity()
        self.k_norm = norm_cls(self.head_dim, **norm_kwargs) if qk_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim)

    def forward(self, hidden_states: torch.Tensor, rope: VisionRotaryEmbeddingFast | None = None) -> torch.Tensor:
        batch_size, sequence_length, channels = hidden_states.shape
        qkv = self.qkv(hidden_states)
        qkv = qkv.view(batch_size, sequence_length, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)

        query = self.q_norm(query)
        key = self.k_norm(key)

        if rope is not None:
            query = rope(query)
            key = rope(key)

        query = query.to(dtype=value.dtype)
        key = key.to(dtype=value.dtype)

        hidden_states = F.scaled_dot_product_attention(query, key, value)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, sequence_length, channels)
        hidden_states = self.proj(hidden_states)
        return hidden_states


class RAEDiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_qknorm: bool = False,
        use_swiglu: bool = True,
        use_rmsnorm: bool = True,
        wo_shift: bool = False,
    ):
        super().__init__()

        if use_rmsnorm:
            self.norm1 = RMSNorm(hidden_size, eps=1e-6)
            self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        else:
            self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.attn = NormAttention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=use_qknorm,
            use_rmsnorm=use_rmsnorm,
        )

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        if use_swiglu:
            self.mlp = SwiGLUFFN(hidden_size, int(2 * mlp_hidden_dim / 3))
        else:
            self.mlp = _ApproximateGELUMLP(hidden_size, mlp_hidden_dim)

        self.wo_shift = wo_shift
        if wo_shift:
            self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 4 * hidden_size, bias=True))
        else:
            self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))

    def forward(
        self,
        hidden_states: torch.Tensor,
        conditioning: torch.Tensor,
        feat_rope: VisionRotaryEmbeddingFast | None = None,
    ) -> torch.Tensor:
        if conditioning.ndim < hidden_states.ndim:
            conditioning = conditioning.unsqueeze(1)

        if self.wo_shift:
            scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(conditioning).chunk(4, dim=-1)
            shift_msa = None
            shift_mlp = None
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(conditioning).chunk(
                6, dim=-1
            )

        hidden_states = hidden_states + _ddt_gate(
            self.attn(_ddt_modulate(self.norm1(hidden_states), shift_msa, scale_msa), rope=feat_rope), gate_msa
        )
        hidden_states = hidden_states + _ddt_gate(
            self.mlp(_ddt_modulate(self.norm2(hidden_states), shift_mlp, scale_mlp)),
            gate_mlp,
        )
        return hidden_states


class RAEDiTFinalLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int, use_rmsnorm: bool = True):
        super().__init__()

        if use_rmsnorm:
            self.norm_final = RMSNorm(hidden_size, eps=1e-6)
        else:
            self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, hidden_states: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        if conditioning.ndim < hidden_states.ndim:
            conditioning = conditioning.unsqueeze(1)

        shift, scale = self.adaLN_modulation(conditioning).chunk(2, dim=-1)
        hidden_states = _ddt_modulate(self.norm_final(hidden_states), shift, scale)
        hidden_states = self.linear(hidden_states)
        return hidden_states


class RAEDiTTransformer2DModel(ModelMixin, ConfigMixin):
    r"""
    Stage-2 latent diffusion transformer used by the RAE paper.

    The architecture mirrors the upstream two-stream `DiTwDDTHead` design:
    an encoder path first builds conditioning tokens from the latent input,
    then a decoder path denoises the latent tokens conditioned on those
    encoded tokens.
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        sample_size: int = 16,
        patch_size: int | tuple[int, int] | list[int] = 1,
        in_channels: int = 768,
        hidden_size: int | tuple[int, int] | list[int] = (1152, 2048),
        depth: int | tuple[int, int] | list[int] = (28, 2),
        num_heads: int | tuple[int, int] | list[int] = (16, 16),
        mlp_ratio: float = 4.0,
        class_dropout_prob: float = 0.1,
        num_classes: int = 1000,
        use_qknorm: bool = False,
        use_swiglu: bool = True,
        use_rope: bool = True,
        use_rmsnorm: bool = True,
        wo_shift: bool = False,
        use_pos_embed: bool = True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = in_channels
        self.gradient_checkpointing = False

        encoder_hidden_size, decoder_hidden_size = _to_pair(hidden_size, "hidden_size")
        encoder_num_layers, decoder_num_layers = _to_pair(depth, "depth")
        encoder_num_attention_heads, decoder_num_attention_heads = _to_pair(num_heads, "num_heads")

        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.num_encoder_blocks = encoder_num_layers
        self.num_decoder_blocks = decoder_num_layers
        self.num_blocks = encoder_num_layers + decoder_num_layers
        self.use_rope = use_rope
        self.use_pos_embed = use_pos_embed

        self.s_patch_size, self.x_patch_size = _to_pair(patch_size, "patch_size")

        self.s_channel_per_token = in_channels * self.s_patch_size * self.s_patch_size
        self.x_channel_per_token = in_channels * self.x_patch_size * self.x_patch_size

        self.s_embedder = PatchEmbed(
            height=sample_size,
            width=sample_size,
            patch_size=self.s_patch_size,
            in_channels=in_channels,
            embed_dim=encoder_hidden_size,
            bias=True,
            pos_embed_type=None,
        )
        self.x_embedder = PatchEmbed(
            height=sample_size,
            width=sample_size,
            patch_size=self.x_patch_size,
            in_channels=in_channels,
            embed_dim=decoder_hidden_size,
            bias=True,
            pos_embed_type=None,
        )

        self.s_projector = (
            nn.Linear(encoder_hidden_size, decoder_hidden_size)
            if encoder_hidden_size != decoder_hidden_size
            else nn.Identity()
        )
        self.t_embedder = GaussianFourierEmbedding(encoder_hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, encoder_hidden_size, class_dropout_prob)
        self.final_layer = RAEDiTFinalLayer(
            decoder_hidden_size,
            out_channels=self.x_channel_per_token,
            use_rmsnorm=use_rmsnorm,
        )

        num_patches = self.s_embedder.height * self.s_embedder.width
        if use_pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, encoder_hidden_size), requires_grad=False)
            self.x_pos_embed = None
        else:
            self.register_parameter("pos_embed", None)
            self.x_pos_embed = None

        if use_rope:
            encoder_rope_dim = encoder_hidden_size // encoder_num_attention_heads // 2
            decoder_rope_dim = decoder_hidden_size // decoder_num_attention_heads // 2
            encoder_side = int(sqrt(num_patches))
            decoder_side = int(sqrt(self.x_embedder.height * self.x_embedder.width))
            self.enc_feat_rope = VisionRotaryEmbeddingFast(encoder_rope_dim, pt_seq_len=encoder_side)
            self.dec_feat_rope = VisionRotaryEmbeddingFast(decoder_rope_dim, pt_seq_len=decoder_side)
        else:
            self.enc_feat_rope = None
            self.dec_feat_rope = None

        self.blocks = nn.ModuleList(
            [
                RAEDiTBlock(
                    hidden_size=encoder_hidden_size if index < encoder_num_layers else decoder_hidden_size,
                    num_heads=encoder_num_attention_heads
                    if index < encoder_num_layers
                    else decoder_num_attention_heads,
                    mlp_ratio=mlp_ratio,
                    use_qknorm=use_qknorm,
                    use_swiglu=use_swiglu,
                    use_rmsnorm=use_rmsnorm,
                    wo_shift=wo_shift,
                )
                for index in range(self.num_blocks)
            ]
        )

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        nn.init.xavier_uniform_(self.x_embedder.proj.weight.view(self.x_embedder.proj.weight.shape[0], -1))
        nn.init.constant_(self.x_embedder.proj.bias, 0)
        nn.init.xavier_uniform_(self.s_embedder.proj.weight.view(self.s_embedder.proj.weight.shape[0], -1))
        nn.init.constant_(self.s_embedder.proj.bias, 0)
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        if self.use_pos_embed:
            pos_embed = get_2d_sincos_pos_embed(
                self.pos_embed.shape[-1], int(sqrt(self.pos_embed.shape[1])), output_type="pt"
            )
            self.pos_embed.data.copy_(pos_embed.float().unsqueeze(0))

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, hidden_states: torch.Tensor) -> torch.Tensor:
        channels = self.in_channels
        patch_size = self.x_embedder.patch_size
        if isinstance(patch_size, tuple):
            patch_size = patch_size[0]

        height = width = int(sqrt(hidden_states.shape[1]))
        if height * width != hidden_states.shape[1]:
            raise ValueError("Sequence length must form a square grid for unpatchify.")

        hidden_states = hidden_states.reshape(hidden_states.shape[0], height, width, patch_size, patch_size, channels)
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        hidden_states = hidden_states.reshape(
            hidden_states.shape[0], channels, height * patch_size, width * patch_size
        )
        return hidden_states

    def _run_block(
        self,
        block: RAEDiTBlock,
        hidden_states: torch.Tensor,
        conditioning: torch.Tensor,
        feat_rope: VisionRotaryEmbeddingFast | None,
    ) -> torch.Tensor:
        if torch.is_grad_enabled() and self.gradient_checkpointing:

            def custom_forward(hidden_states: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
                return block(hidden_states, conditioning, feat_rope=feat_rope)

            return self._gradient_checkpointing_func(custom_forward, hidden_states, conditioning)
        return block(hidden_states, conditioning, feat_rope=feat_rope)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor | None = None,
        class_labels: torch.LongTensor | None = None,
        conditioning_hidden_states: torch.Tensor | None = None,
        return_dict: bool = True,
    ) -> Transformer2DModelOutput | tuple[torch.Tensor]:
        if timestep is None:
            raise ValueError("`timestep` must be provided.")
        if class_labels is None:
            raise ValueError("`class_labels` must be provided.")

        timestep = timestep.reshape(-1).to(hidden_states.device)
        class_labels = class_labels.reshape(-1).to(hidden_states.device)

        timestep_emb = self.t_embedder(timestep)
        class_emb = self.y_embedder(class_labels, self.training)
        conditioning = F.silu(timestep_emb + class_emb)

        if conditioning_hidden_states is None:
            conditioning_hidden_states = self.s_embedder(hidden_states)
            if self.use_pos_embed:
                conditioning_hidden_states = conditioning_hidden_states + self.pos_embed

            for block_idx in range(self.num_encoder_blocks):
                conditioning_hidden_states = self._run_block(
                    self.blocks[block_idx],
                    conditioning_hidden_states,
                    conditioning,
                    self.enc_feat_rope,
                )

            conditioning_hidden_states = F.silu(timestep_emb.unsqueeze(1) + conditioning_hidden_states)

        conditioning_hidden_states = self.s_projector(conditioning_hidden_states)

        hidden_states = self.x_embedder(hidden_states)
        if self.use_pos_embed and self.x_pos_embed is not None:
            hidden_states = hidden_states + self.x_pos_embed

        for block_idx in range(self.num_encoder_blocks, self.num_blocks):
            hidden_states = self._run_block(
                self.blocks[block_idx],
                hidden_states,
                conditioning_hidden_states,
                self.dec_feat_rope,
            )

        hidden_states = self.final_layer(hidden_states, conditioning_hidden_states)
        hidden_states = self.unpatchify(hidden_states)

        if not return_dict:
            return (hidden_states,)

        return Transformer2DModelOutput(sample=hidden_states)
