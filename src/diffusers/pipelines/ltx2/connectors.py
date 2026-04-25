import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders import PeftAdapterMixin
from ...models.attention import FeedForward
from ...models.modeling_utils import ModelMixin
from ...models.transformers.transformer_ltx2 import LTX2Attention, LTX2AudioVideoAttnProcessor


def per_layer_masked_mean_norm(
    text_hidden_states: torch.Tensor,
    sequence_lengths: torch.Tensor,
    device: str | torch.device,
    padding_side: str = "left",
    scale_factor: int = 8,
    eps: float = 1e-6,
):
    """
    Performs per-batch per-layer normalization using a masked mean and range on per-layer text encoder hidden_states.
    Respects the padding of the hidden states.

    Args:
        text_hidden_states (`torch.Tensor` of shape `(batch_size, seq_len, hidden_dim, num_layers)`):
            Per-layer hidden_states from a text encoder (e.g. `Gemma3ForConditionalGeneration`).
        sequence_lengths (`torch.Tensor of shape `(batch_size,)`):
            The number of valid (non-padded) tokens for each batch instance.
        device: (`str` or `torch.device`, *optional*):
            torch device to place the resulting embeddings on
        padding_side: (`str`, *optional*, defaults to `"left"`):
            Whether the text tokenizer performs padding on the `"left"` or `"right"`.
        scale_factor (`int`, *optional*, defaults to `8`):
            Scaling factor to multiply the normalized hidden states by.
        eps (`float`, *optional*, defaults to `1e-6`):
            A small positive value for numerical stability when performing normalization.

    Returns:
        `torch.Tensor` of shape `(batch_size, seq_len, hidden_dim * num_layers)`:
            Normed and flattened text encoder hidden states.
    """
    batch_size, seq_len, hidden_dim, num_layers = text_hidden_states.shape
    original_dtype = text_hidden_states.dtype

    # Create padding mask
    token_indices = torch.arange(seq_len, device=device).unsqueeze(0)
    if padding_side == "right":
        # For right padding, valid tokens are from 0 to sequence_length-1
        mask = token_indices < sequence_lengths[:, None]  # [batch_size, seq_len]
    elif padding_side == "left":
        # For left padding, valid tokens are from (T - sequence_length) to T-1
        start_indices = seq_len - sequence_lengths[:, None]  # [batch_size, 1]
        mask = token_indices >= start_indices  # [B, T]
    else:
        raise ValueError(f"padding_side must be 'left' or 'right', got {padding_side}")
    mask = mask[:, :, None, None]  # [batch_size, seq_len] --> [batch_size, seq_len, 1, 1]

    # Compute masked mean over non-padding positions of shape (batch_size, 1, 1, seq_len)
    masked_text_hidden_states = text_hidden_states.masked_fill(~mask, 0.0)
    num_valid_positions = (sequence_lengths * hidden_dim).view(batch_size, 1, 1, 1)
    masked_mean = masked_text_hidden_states.sum(dim=(1, 2), keepdim=True) / (num_valid_positions + eps)

    # Compute min/max over non-padding positions of shape (batch_size, 1, 1 seq_len)
    x_min = text_hidden_states.masked_fill(~mask, float("inf")).amin(dim=(1, 2), keepdim=True)
    x_max = text_hidden_states.masked_fill(~mask, float("-inf")).amax(dim=(1, 2), keepdim=True)

    # Normalization
    normalized_hidden_states = (text_hidden_states - masked_mean) / (x_max - x_min + eps)
    normalized_hidden_states = normalized_hidden_states * scale_factor

    # Pack the hidden states to a 3D tensor (batch_size, seq_len, hidden_dim * num_layers)
    normalized_hidden_states = normalized_hidden_states.flatten(2)
    mask_flat = mask.squeeze(-1).expand(-1, -1, hidden_dim * num_layers)
    normalized_hidden_states = normalized_hidden_states.masked_fill(~mask_flat, 0.0)
    normalized_hidden_states = normalized_hidden_states.to(dtype=original_dtype)
    return normalized_hidden_states


def per_token_rms_norm(text_encoder_hidden_states: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    variance = torch.mean(text_encoder_hidden_states**2, dim=2, keepdim=True)
    norm_text_encoder_hidden_states = text_encoder_hidden_states * torch.rsqrt(variance + eps)
    return norm_text_encoder_hidden_states


class LTX2RotaryPosEmbed1d(nn.Module):
    """
    1D rotary positional embeddings (RoPE) for the LTX 2.0 text encoder connectors.
    """

    def __init__(
        self,
        dim: int,
        base_seq_len: int = 4096,
        theta: float = 10000.0,
        double_precision: bool = True,
        rope_type: str = "interleaved",
        num_attention_heads: int = 32,
    ):
        super().__init__()
        if rope_type not in ["interleaved", "split"]:
            raise ValueError(f"{rope_type=} not supported. Choose between 'interleaved' and 'split'.")

        self.dim = dim
        self.base_seq_len = base_seq_len
        self.theta = theta
        self.double_precision = double_precision
        self.rope_type = rope_type
        self.num_attention_heads = num_attention_heads

    def forward(
        self,
        batch_size: int,
        pos: int,
        device: str | torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 1. Get 1D position ids
        grid_1d = torch.arange(pos, dtype=torch.float32, device=device)
        # Get fractional indices relative to self.base_seq_len
        grid_1d = grid_1d / self.base_seq_len
        grid = grid_1d.unsqueeze(0).repeat(batch_size, 1)  # [batch_size, seq_len]

        # 2. Calculate 1D RoPE frequencies
        num_rope_elems = 2  # 1 (because 1D) * 2 (for cos, sin) = 2
        freqs_dtype = torch.float64 if self.double_precision else torch.float32
        pow_indices = torch.pow(
            self.theta,
            torch.linspace(start=0.0, end=1.0, steps=self.dim // num_rope_elems, dtype=freqs_dtype, device=device),
        )
        freqs = (pow_indices * torch.pi / 2.0).to(dtype=torch.float32)

        # 3. Matrix-vector outer product between pos ids of shape (batch_size, seq_len) and freqs vector of shape
        # (self.dim // 2,).
        freqs = (grid.unsqueeze(-1) * 2 - 1) * freqs  # [B, seq_len, self.dim // 2]

        # 4. Get real, interleaved (cos, sin) frequencies, padded to self.dim
        if self.rope_type == "interleaved":
            cos_freqs = freqs.cos().repeat_interleave(2, dim=-1)
            sin_freqs = freqs.sin().repeat_interleave(2, dim=-1)

            if self.dim % num_rope_elems != 0:
                cos_padding = torch.ones_like(cos_freqs[:, :, : self.dim % num_rope_elems])
                sin_padding = torch.zeros_like(sin_freqs[:, :, : self.dim % num_rope_elems])
                cos_freqs = torch.cat([cos_padding, cos_freqs], dim=-1)
                sin_freqs = torch.cat([sin_padding, sin_freqs], dim=-1)

        elif self.rope_type == "split":
            expected_freqs = self.dim // 2
            current_freqs = freqs.shape[-1]
            pad_size = expected_freqs - current_freqs
            cos_freq = freqs.cos()
            sin_freq = freqs.sin()

            if pad_size != 0:
                cos_padding = torch.ones_like(cos_freq[:, :, :pad_size])
                sin_padding = torch.zeros_like(sin_freq[:, :, :pad_size])

                cos_freq = torch.concatenate([cos_padding, cos_freq], axis=-1)
                sin_freq = torch.concatenate([sin_padding, sin_freq], axis=-1)

            # Reshape freqs to be compatible with multi-head attention
            b = cos_freq.shape[0]
            t = cos_freq.shape[1]

            cos_freq = cos_freq.reshape(b, t, self.num_attention_heads, -1)
            sin_freq = sin_freq.reshape(b, t, self.num_attention_heads, -1)

            cos_freqs = torch.swapaxes(cos_freq, 1, 2)  # (B,H,T,D//2)
            sin_freqs = torch.swapaxes(sin_freq, 1, 2)  # (B,H,T,D//2)

        return cos_freqs, sin_freqs


class LTX2TransformerBlock1d(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        activation_fn: str = "gelu-approximate",
        eps: float = 1e-6,
        rope_type: str = "interleaved",
        apply_gated_attention: bool = False,
    ):
        super().__init__()

        self.norm1 = torch.nn.RMSNorm(dim, eps=eps, elementwise_affine=False)
        self.attn1 = LTX2Attention(
            query_dim=dim,
            heads=num_attention_heads,
            kv_heads=num_attention_heads,
            dim_head=attention_head_dim,
            rope_type=rope_type,
            apply_gated_attention=apply_gated_attention,
            processor=LTX2AudioVideoAttnProcessor(),
        )

        self.norm2 = torch.nn.RMSNorm(dim, eps=eps, elementwise_affine=False)
        self.ff = FeedForward(dim, activation_fn=activation_fn)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        rotary_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        norm_hidden_states = self.norm1(hidden_states)
        attn_hidden_states = self.attn1(norm_hidden_states, attention_mask=attention_mask, query_rotary_emb=rotary_emb)
        hidden_states = hidden_states + attn_hidden_states

        norm_hidden_states = self.norm2(hidden_states)
        ff_hidden_states = self.ff(norm_hidden_states)
        hidden_states = hidden_states + ff_hidden_states

        return hidden_states


class LTX2ConnectorTransformer1d(nn.Module):
    """
    A 1D sequence transformer for modalities such as text.

    In LTX 2.0, this is used to process the text encoder hidden states for each of the video and audio streams.
    """

    _supports_gradient_checkpointing = True

    def __init__(
        self,
        num_attention_heads: int = 30,
        attention_head_dim: int = 128,
        num_layers: int = 2,
        num_learnable_registers: int | None = 128,
        rope_base_seq_len: int = 4096,
        rope_theta: float = 10000.0,
        rope_double_precision: bool = True,
        eps: float = 1e-6,
        causal_temporal_positioning: bool = False,
        rope_type: str = "interleaved",
        gated_attention: bool = False,
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.inner_dim = num_attention_heads * attention_head_dim
        self.causal_temporal_positioning = causal_temporal_positioning

        self.num_learnable_registers = num_learnable_registers
        self.learnable_registers = None
        if num_learnable_registers is not None:
            init_registers = torch.rand(num_learnable_registers, self.inner_dim) * 2.0 - 1.0
            self.learnable_registers = torch.nn.Parameter(init_registers)

        self.rope = LTX2RotaryPosEmbed1d(
            self.inner_dim,
            base_seq_len=rope_base_seq_len,
            theta=rope_theta,
            double_precision=rope_double_precision,
            rope_type=rope_type,
            num_attention_heads=num_attention_heads,
        )

        self.transformer_blocks = torch.nn.ModuleList(
            [
                LTX2TransformerBlock1d(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    rope_type=rope_type,
                    apply_gated_attention=gated_attention,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm_out = torch.nn.RMSNorm(self.inner_dim, eps=eps, elementwise_affine=False)

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        attn_mask_binarize_threshold: float = -9000.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # hidden_states shape: [batch_size, seq_len, hidden_dim]
        # attention_mask shape: [batch_size, seq_len] or [batch_size, 1, 1, seq_len]
        batch_size, seq_len, _ = hidden_states.shape

        # 1. Replace padding with learned registers, if using
        if self.learnable_registers is not None:
            if seq_len % self.num_learnable_registers != 0:
                raise ValueError(
                    f"The `hidden_states` sequence length {hidden_states.shape[1]} should be divisible by the number"
                    f" of learnable registers {self.num_learnable_registers}"
                )

            num_register_repeats = seq_len // self.num_learnable_registers
            registers = torch.tile(self.learnable_registers, (num_register_repeats, 1))  # [seq_len, inner_dim]

            binary_attn_mask = (attention_mask >= attn_mask_binarize_threshold).int()
            if binary_attn_mask.ndim == 4:
                binary_attn_mask = binary_attn_mask.squeeze(1).squeeze(1)  # [B, 1, 1, L] --> [B, L]

            hidden_states_non_padded = [hidden_states[i, binary_attn_mask[i].bool(), :] for i in range(batch_size)]
            valid_seq_lens = [x.shape[0] for x in hidden_states_non_padded]
            pad_lengths = [seq_len - valid_seq_len for valid_seq_len in valid_seq_lens]
            padded_hidden_states = [
                F.pad(x, pad=(0, 0, 0, p), value=0) for x, p in zip(hidden_states_non_padded, pad_lengths)
            ]
            padded_hidden_states = torch.cat([x.unsqueeze(0) for x in padded_hidden_states], dim=0)  # [B, L, D]

            flipped_mask = torch.flip(binary_attn_mask, dims=[1]).unsqueeze(-1)  # [B, L, 1]
            hidden_states = flipped_mask * padded_hidden_states + (1 - flipped_mask) * registers

            # Overwrite attention_mask with an all-zeros mask if using registers.
            attention_mask = torch.zeros_like(attention_mask)

        # 2. Calculate 1D RoPE positional embeddings
        rotary_emb = self.rope(batch_size, seq_len, device=hidden_states.device)

        # 3. Run 1D transformer blocks
        for block in self.transformer_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(block, hidden_states, attention_mask, rotary_emb)
            else:
                hidden_states = block(hidden_states, attention_mask=attention_mask, rotary_emb=rotary_emb)

        hidden_states = self.norm_out(hidden_states)

        return hidden_states, attention_mask


class LTX2TextConnectors(ModelMixin, PeftAdapterMixin, ConfigMixin):
    """
    Text connector stack used by LTX 2.0 to process the packed text encoder hidden states for both the video and audio
    streams.
    """

    @register_to_config
    def __init__(
        self,
        caption_channels: int = 3840,  # default Gemma-3-12B text encoder hidden_size
        text_proj_in_factor: int = 49,  # num_layers + 1 for embedding layer = 48 + 1 for Gemma-3-12B
        video_connector_num_attention_heads: int = 30,
        video_connector_attention_head_dim: int = 128,
        video_connector_num_layers: int = 2,
        video_connector_num_learnable_registers: int | None = 128,
        video_gated_attn: bool = False,
        audio_connector_num_attention_heads: int = 30,
        audio_connector_attention_head_dim: int = 128,
        audio_connector_num_layers: int = 2,
        audio_connector_num_learnable_registers: int | None = 128,
        audio_gated_attn: bool = False,
        connector_rope_base_seq_len: int = 4096,
        rope_theta: float = 10000.0,
        rope_double_precision: bool = True,
        causal_temporal_positioning: bool = False,
        rope_type: str = "interleaved",
        per_modality_projections: bool = False,
        video_hidden_dim: int = 4096,
        audio_hidden_dim: int = 2048,
        proj_bias: bool = False,
    ):
        super().__init__()
        text_encoder_dim = caption_channels * text_proj_in_factor
        if per_modality_projections:
            self.video_text_proj_in = nn.Linear(text_encoder_dim, video_hidden_dim, bias=proj_bias)
            self.audio_text_proj_in = nn.Linear(text_encoder_dim, audio_hidden_dim, bias=proj_bias)
        else:
            self.text_proj_in = nn.Linear(text_encoder_dim, caption_channels, bias=proj_bias)

        self.video_connector = LTX2ConnectorTransformer1d(
            num_attention_heads=video_connector_num_attention_heads,
            attention_head_dim=video_connector_attention_head_dim,
            num_layers=video_connector_num_layers,
            num_learnable_registers=video_connector_num_learnable_registers,
            rope_base_seq_len=connector_rope_base_seq_len,
            rope_theta=rope_theta,
            rope_double_precision=rope_double_precision,
            causal_temporal_positioning=causal_temporal_positioning,
            rope_type=rope_type,
            gated_attention=video_gated_attn,
        )
        self.audio_connector = LTX2ConnectorTransformer1d(
            num_attention_heads=audio_connector_num_attention_heads,
            attention_head_dim=audio_connector_attention_head_dim,
            num_layers=audio_connector_num_layers,
            num_learnable_registers=audio_connector_num_learnable_registers,
            rope_base_seq_len=connector_rope_base_seq_len,
            rope_theta=rope_theta,
            rope_double_precision=rope_double_precision,
            causal_temporal_positioning=causal_temporal_positioning,
            rope_type=rope_type,
            gated_attention=audio_gated_attn,
        )

    def forward(
        self,
        text_encoder_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        padding_side: str = "left",
        scale_factor: int = 8,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Given per-layer text encoder hidden_states, extracts features and runs per-modality connectors to get text
        embeddings for the LTX-2.X DiT models.

        Args:
            text_encoder_hidden_states (`torch.Tensor`)):
                Per-layer text encoder hidden_states. Can either be 4D with shape `(batch_size, seq_len,
                caption_channels, text_proj_in_factor) or 3D with the last two dimensions flattened.
            attention_mask (`torch.Tensor` of shape `(batch_size, seq_len)`):
                Multiplicative binary attention mask where 1s indicate unmasked positions and 0s indicate masked
                positions.
            padding_side (`str`, *optional*, defaults to `"left"`):
                The padding side used by the text encoder's text encoder (either `"left"` or `"right"`). Defaults to
                `"left"` as this is what the default Gemma3-12B text encoder uses. Only used if
                `per_modality_projections` is `False` (LTX-2.0 models).
            scale_factor (`int`, *optional*, defaults to `8`):
                Scale factor for masked mean/range normalization. Only used if `per_modality_projections` is `False`
                (LTX-2.0 models).
        """
        if text_encoder_hidden_states.ndim == 3:
            # Ensure shape is [batch_size, seq_len, caption_channels, text_proj_in_factor]
            text_encoder_hidden_states = text_encoder_hidden_states.unflatten(2, (self.config.caption_channels, -1))

        if self.config.per_modality_projections:
            # LTX-2.3
            norm_text_encoder_hidden_states = per_token_rms_norm(text_encoder_hidden_states)

            norm_text_encoder_hidden_states = norm_text_encoder_hidden_states.flatten(2, 3)
            bool_mask = attention_mask.bool().unsqueeze(-1)
            norm_text_encoder_hidden_states = torch.where(
                bool_mask, norm_text_encoder_hidden_states, torch.zeros_like(norm_text_encoder_hidden_states)
            )

            # Rescale norms with respect to video and audio dims for feature extractors
            video_scale_factor = math.sqrt(self.config.video_hidden_dim / self.config.caption_channels)
            video_norm_text_emb = norm_text_encoder_hidden_states * video_scale_factor
            audio_scale_factor = math.sqrt(self.config.audio_hidden_dim / self.config.caption_channels)
            audio_norm_text_emb = norm_text_encoder_hidden_states * audio_scale_factor

            # Per-Modality Feature extractors
            video_text_emb_proj = self.video_text_proj_in(video_norm_text_emb)
            audio_text_emb_proj = self.audio_text_proj_in(audio_norm_text_emb)
        else:
            # LTX-2.0
            sequence_lengths = attention_mask.sum(dim=-1)
            norm_text_encoder_hidden_states = per_layer_masked_mean_norm(
                text_hidden_states=text_encoder_hidden_states,
                sequence_lengths=sequence_lengths,
                device=text_encoder_hidden_states.device,
                padding_side=padding_side,
                scale_factor=scale_factor,
            )

            text_emb_proj = self.text_proj_in(norm_text_encoder_hidden_states)
            video_text_emb_proj = text_emb_proj
            audio_text_emb_proj = text_emb_proj

        # Convert to additive attention mask for connectors
        text_dtype = video_text_emb_proj.dtype
        attention_mask = (attention_mask.to(torch.int64) - 1).to(text_dtype)
        attention_mask = attention_mask.reshape(attention_mask.shape[0], 1, -1, attention_mask.shape[-1])
        add_attn_mask = attention_mask * torch.finfo(text_dtype).max

        video_text_embedding, video_attn_mask = self.video_connector(video_text_emb_proj, add_attn_mask)

        # Convert video attn mask to binary (multiplicative) mask and mask video text embedding
        binary_attn_mask = (video_attn_mask < 1e-6).to(torch.int64)
        binary_attn_mask = binary_attn_mask.reshape(video_text_embedding.shape[0], video_text_embedding.shape[1], 1)
        video_text_embedding = video_text_embedding * binary_attn_mask

        audio_text_embedding, _ = self.audio_connector(audio_text_emb_proj, add_attn_mask)

        return video_text_embedding, audio_text_embedding, binary_attn_mask.squeeze(-1)
