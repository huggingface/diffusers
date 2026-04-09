import math
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models import ModelMixin
from diffusers.models.attention import AttentionModuleMixin, FeedForward
from diffusers.models.attention_dispatch import AttentionBackendName, dispatch_attention_fn
from diffusers.models.embeddings import (
    PixArtAlphaTextProjection,
    TimestepEmbedding,
    Timesteps,
    apply_rotary_emb,
    get_1d_rotary_pos_embed,
)
from diffusers.models.normalization import RMSNorm


def _create_modulation(
        modulate_type: str,
        hidden_size: int,
        factor: int,
        dtype=None,
        device=None):
    factory_kwargs = {"dtype": dtype, "device": device}
    if modulate_type == 'wanx':
        return _WanModulation(hidden_size, factor, **factory_kwargs)
    raise ValueError(
        f"Unknown modulation type: {modulate_type}. Only 'wanx' is supported.")


class _WanModulation(nn.Module):
    """Modulation layer for WanX."""

    def __init__(
        self,
        hidden_size: int,
        factor: int,
        dtype=None,
        device=None,
    ):
        super().__init__()
        self.factor = factor
        self.modulate_table = nn.Parameter(
            torch.zeros(1, factor, hidden_size,
                        dtype=dtype, device=device) / hidden_size**0.5,
            requires_grad=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) != 3:
            x = x.unsqueeze(1)
        return [o.squeeze(1) for o in (self.modulate_table + x).chunk(self.factor, dim=1)]


class JoyAIJointAttnProcessor:
    _attention_backend = None
    _parallel_config = None

    def __call__(
        self,
        attn: "JoyAIJointAttention",
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        attention_kwargs: Optional[dict[str, Any]] = None,
    ) -> torch.Tensor:
        attention_kwargs = attention_kwargs or {}
        backend = AttentionBackendName.NATIVE if attn.backend == "torch_spda" else AttentionBackendName.FLASH_VARLEN

        try:
            return dispatch_attention_fn(
                query,
                key,
                value,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False,
                attention_kwargs=attention_kwargs,
                backend=backend,
                parallel_config=self._parallel_config,
            )
        except (RuntimeError, ValueError, TypeError):
            return dispatch_attention_fn(
                query,
                key,
                value,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False,
                attention_kwargs=attention_kwargs,
                backend=AttentionBackendName.NATIVE,
                parallel_config=self._parallel_config,
            )


class JoyAIJointAttention(nn.Module, AttentionModuleMixin):
    _default_processor_cls = JoyAIJointAttnProcessor
    _available_processors = [JoyAIJointAttnProcessor]

    def __init__(self, backend: str = "flash_attn", processor=None) -> None:
        super().__init__()
        self.backend = backend
        if processor is None:
            processor = self._default_processor_cls()
        self.set_processor(processor)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        attention_kwargs: Optional[dict[str, Any]] = None,
    ) -> torch.Tensor:
        return self.processor(self, query, key, value, attention_mask, attention_kwargs)


class JoyAIImageTransformerBlock(nn.Module):
    """Joint text-image transformer block."""

    def __init__(
        self,
        hidden_size: int,
        heads_num: int,
        mlp_width_ratio: float,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        dit_modulation_type: Optional[str] = "wanx",
        attn_backend: str = 'flash_attn',
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.dit_modulation_type = dit_modulation_type
        self.heads_num = heads_num
        head_dim = hidden_size // heads_num
        mlp_hidden_dim = int(hidden_size * mlp_width_ratio)

        self.img_mod = _create_modulation(
            modulate_type=self.dit_modulation_type,
            hidden_size=hidden_size,
            factor=6,
            **factory_kwargs,
        )
        self.img_norm1 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs
        )

        self.img_attn_qkv = nn.Linear(
            hidden_size, hidden_size * 3, bias=True, **factory_kwargs
        )
        self.img_attn_q_norm = RMSNorm(head_dim, elementwise_affine=True, eps=1e-6)
        self.img_attn_k_norm = RMSNorm(head_dim, elementwise_affine=True, eps=1e-6)
        self.img_attn_proj = nn.Linear(
            hidden_size, hidden_size, bias=True, **factory_kwargs
        )

        self.img_norm2 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs
        )
        self.img_mlp = FeedForward(hidden_size, inner_dim=mlp_hidden_dim,
                                   activation_fn="gelu-approximate")

        self.txt_mod = _create_modulation(
            modulate_type=self.dit_modulation_type,
            hidden_size=hidden_size,
            factor=6,
            **factory_kwargs,
        )
        self.txt_norm1 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs
        )

        self.txt_attn_qkv = nn.Linear(
            hidden_size, hidden_size * 3, bias=True, **factory_kwargs
        )
        self.txt_attn_q_norm = RMSNorm(head_dim, elementwise_affine=True, eps=1e-6)
        self.txt_attn_k_norm = RMSNorm(head_dim, elementwise_affine=True, eps=1e-6)
        self.txt_attn_proj = nn.Linear(
            hidden_size, hidden_size, bias=True, **factory_kwargs
        )
        self.attn = JoyAIJointAttention(attn_backend)

        self.txt_norm2 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs
        )
        self.txt_mlp = FeedForward(hidden_size, inner_dim=mlp_hidden_dim,
                                   activation_fn="gelu-approximate")

    @staticmethod
    def _modulate(
        hidden_states: torch.Tensor, shift: torch.Tensor | None = None, scale: torch.Tensor | None = None
    ) -> torch.Tensor:
        if scale is None and shift is None:
            return hidden_states
        if shift is None:
            return hidden_states * (1 + scale.unsqueeze(1))
        if scale is None:
            return hidden_states + shift.unsqueeze(1)
        return hidden_states * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    @staticmethod
    def _apply_gate(
        hidden_states: torch.Tensor, gate: torch.Tensor | None = None, tanh: bool = False
    ) -> torch.Tensor:
        if gate is None:
            return hidden_states
        if tanh:
            return hidden_states * gate.unsqueeze(1).tanh()
        return hidden_states * gate.unsqueeze(1)

    def forward(
        self,
        img: torch.Tensor,
        txt: torch.Tensor,
        vec: torch.Tensor,
        vis_freqs_cis: tuple = None,
        txt_freqs_cis: tuple = None,
        attn_kwargs: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        (
            img_mod1_shift,
            img_mod1_scale,
            img_mod1_gate,
            img_mod2_shift,
            img_mod2_scale,
            img_mod2_gate,
        ) = self.img_mod(vec)
        (
            txt_mod1_shift,
            txt_mod1_scale,
            txt_mod1_gate,
            txt_mod2_shift,
            txt_mod2_scale,
            txt_mod2_gate,
        ) = self.txt_mod(vec)

        img_modulated = self.img_norm1(img)
        img_modulated = self._modulate(
            img_modulated, shift=img_mod1_shift, scale=img_mod1_scale
        )
        img_qkv = self.img_attn_qkv(img_modulated)
        batch_size, image_sequence_length, _ = img_qkv.shape
        img_qkv = img_qkv.view(batch_size, image_sequence_length, 3, self.heads_num, -1).permute(2, 0, 1, 3, 4)
        img_q, img_k, img_v = img_qkv.unbind(0)
        img_q = self.img_attn_q_norm(img_q).to(img_v)
        img_k = self.img_attn_k_norm(img_k).to(img_v)

        if vis_freqs_cis is not None:
            img_q = apply_rotary_emb(img_q, vis_freqs_cis, sequence_dim=1)
            img_k = apply_rotary_emb(img_k, vis_freqs_cis, sequence_dim=1)

        txt_modulated = self.txt_norm1(txt)
        txt_modulated = self._modulate(
            txt_modulated, shift=txt_mod1_shift, scale=txt_mod1_scale
        )
        txt_qkv = self.txt_attn_qkv(txt_modulated)
        _, text_sequence_length, _ = txt_qkv.shape
        txt_qkv = txt_qkv.view(batch_size, text_sequence_length, 3, self.heads_num, -1).permute(2, 0, 1, 3, 4)
        txt_q, txt_k, txt_v = txt_qkv.unbind(0)
        txt_q = self.txt_attn_q_norm(txt_q).to(txt_v)
        txt_k = self.txt_attn_k_norm(txt_k).to(txt_v)

        if txt_freqs_cis is not None:
            raise NotImplementedError("RoPE text is not supported for inference")


        attention_output = self.attn(
            torch.cat((img_q, txt_q), dim=1),
            torch.cat((img_k, txt_k), dim=1),
            torch.cat((img_v, txt_v), dim=1),
            attention_mask=attn_kwargs.get("attention_mask") if attn_kwargs is not None else None,
            attention_kwargs=attn_kwargs,
        )
        attention_output = attention_output.flatten(2, 3)
        image_attention_output = attention_output[:, : img.shape[1]]
        text_attention_output = attention_output[:, img.shape[1]:]

        img = img + self._apply_gate(self.img_attn_proj(image_attention_output),
                               gate=img_mod1_gate)
        img = img + self._apply_gate(
            self.img_mlp(
                self._modulate(
                    self.img_norm2(img), shift=img_mod2_shift, scale=img_mod2_scale
                )
            ),
            gate=img_mod2_gate,
        )

        txt = txt + self._apply_gate(self.txt_attn_proj(text_attention_output),
                               gate=txt_mod1_gate)
        txt = txt + self._apply_gate(
            self.txt_mlp(
                self._modulate(
                    self.txt_norm2(txt), shift=txt_mod2_shift, scale=txt_mod2_scale
                )
            ),
            gate=txt_mod2_gate,
        )

        return img, txt


class JoyAITimeTextEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        time_freq_dim: int,
        time_proj_dim: int,
        text_embed_dim: int,
    ):
        super().__init__()

        self.timesteps_proj = Timesteps(
            num_channels=time_freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedder = TimestepEmbedding(
            in_channels=time_freq_dim, time_embed_dim=dim)
        self.act_fn = nn.SiLU()
        self.time_proj = nn.Linear(dim, time_proj_dim)
        self.text_embedder = PixArtAlphaTextProjection(
            text_embed_dim, dim, act_fn="gelu_tanh")

    def forward(
        self,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        timestep = self.timesteps_proj(timestep)

        time_embedder_dtype = next(iter(self.time_embedder.parameters())).dtype
        if timestep.dtype != time_embedder_dtype and time_embedder_dtype != torch.int8:
            timestep = timestep.to(time_embedder_dtype)
        timestep_embedding = self.time_embedder(timestep).type_as(encoder_hidden_states)
        modulation_states = self.time_proj(self.act_fn(timestep_embedding))
        encoder_hidden_states = self.text_embedder(encoder_hidden_states)

        return modulation_states, encoder_hidden_states


class JoyAIImageTransformer3DModel(ModelMixin, ConfigMixin):
    _fsdp_shard_conditions: list = [
        lambda name, module: isinstance(module, JoyAIImageTransformerBlock)]
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        patch_size: tuple[int, int, int] = (1, 2, 2),
        in_channels: int = 4,
        out_channels: int = None,
        hidden_size: int = 3072,
        heads_num: int = 24,
        text_states_dim: int = 4096,
        mlp_width_ratio: float = 4.0,
        mm_double_blocks_depth: int = 20,
        rope_dim_list: tuple[int, int, int] = (16, 56, 56),
        rope_type: str = 'rope',
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        dit_modulation_type: str = "wanx",
        attn_backend: str = 'flash_attn',
        theta: int = 256,
    ):
        self.out_channels = out_channels or in_channels
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.heads_num = heads_num
        self.rope_dim_list = rope_dim_list
        self.dit_modulation_type = dit_modulation_type
        self.rope_type = rope_type
        self.theta = theta

        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        if hidden_size % heads_num != 0:
            raise ValueError(
                f"Hidden size {hidden_size} must be divisible by heads_num {heads_num}"
            )

        self.img_in = nn.Conv3d(
            in_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

        self.condition_embedder = JoyAITimeTextEmbedding(
            dim=hidden_size,
            time_freq_dim=256,
            time_proj_dim=hidden_size * 6,
            text_embed_dim=text_states_dim,
        )

        self.double_blocks = nn.ModuleList(
            [
                JoyAIImageTransformerBlock(
                    self.hidden_size,
                    self.heads_num,
                    mlp_width_ratio=mlp_width_ratio,
                    dit_modulation_type=self.dit_modulation_type,
                    attn_backend=attn_backend,
                    **factory_kwargs,
                )
                for _ in range(mm_double_blocks_depth)
            ]
        )

        self.norm_out = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6
        )
        self.proj_out = nn.Linear(
            hidden_size, out_channels * math.prod(patch_size),
            **factory_kwargs)


    @staticmethod
    def _get_meshgrid_nd(start, *args, dim=2):
        """Build an N-D meshgrid from integer sizes or ranges."""

        def as_tuple(value):
            if isinstance(value, int):
                return (value,) * dim
            if len(value) == dim:
                return value
            raise ValueError(f"Expected length {dim} or int, but got {value}")
        if len(args) == 0:
            num = as_tuple(start)
            start = (0,) * dim
            stop = num
        elif len(args) == 1:
            start = as_tuple(start)
            stop = as_tuple(args[0])
            num = [stop[i] - start[i] for i in range(dim)]
        elif len(args) == 2:
            start = as_tuple(start)
            stop = as_tuple(args[0])
            num = as_tuple(args[1])
        else:
            raise ValueError(f"len(args) should be 0, 1 or 2, but got {len(args)}")

        axis_grid = []
        for i in range(dim):
            a, b, n = start[i], stop[i], num[i]
            g = torch.linspace(a, b, n + 1, dtype=torch.float32)[:n]
            axis_grid.append(g)
        grid = torch.meshgrid(*axis_grid, indexing="ij")
        grid = torch.stack(grid, dim=0)

        return grid


    @staticmethod
    def _get_nd_rotary_pos_embed(
        rope_dim_list,
        start,
        *args,
        theta=10000.0,
        use_real=False,
        text_sequence_length=None,
    ):
        """Build visual and optional text rotary embeddings."""

        grid = JoyAIImageTransformer3DModel._get_meshgrid_nd(
            start, *args, dim=len(rope_dim_list)
        )

        embs = []
        for i in range(len(rope_dim_list)):
            emb = get_1d_rotary_pos_embed(
                rope_dim_list[i],
                grid[i].reshape(-1),
                theta=theta,
                use_real=use_real,
            )
            embs.append(emb)

        if use_real:
            cos = torch.cat([emb[0] for emb in embs], dim=1)
            sin = torch.cat([emb[1] for emb in embs], dim=1)
            vis_emb = (cos, sin)
        else:
            vis_emb = torch.cat(embs, dim=1)
        if text_sequence_length is not None:
            embs_txt = []
            vis_max_ids = grid.view(-1).max().item()
            text_positions = torch.arange(text_sequence_length) + vis_max_ids + 1
            for i in range(len(rope_dim_list)):
                emb = get_1d_rotary_pos_embed(
                    rope_dim_list[i],
                    text_positions,
                    theta=theta,
                    use_real=use_real,
                )
                embs_txt.append(emb)
            if use_real:
                cos = torch.cat([emb[0] for emb in embs_txt], dim=1)
                sin = torch.cat([emb[1] for emb in embs_txt], dim=1)
                txt_emb = (cos, sin)
            else:
                txt_emb = torch.cat(embs_txt, dim=1)
        else:
            txt_emb = None
        return vis_emb, txt_emb




    def get_rotary_pos_embed(self, image_grid_size, text_sequence_length=None):
        target_ndim = 3

        if len(image_grid_size) != target_ndim:
            image_grid_size = [1] * (target_ndim - len(image_grid_size)) + image_grid_size
        head_dim = self.hidden_size // self.heads_num
        rope_dim_list = self.rope_dim_list
        if rope_dim_list is None:
            rope_dim_list = [head_dim //
                             target_ndim for _ in range(target_ndim)]
        assert (
            sum(rope_dim_list) == head_dim
        ), "sum(rope_dim_list) should equal to head_dim of attention layer"
        image_rotary_emb, text_rotary_emb = self._get_nd_rotary_pos_embed(
            rope_dim_list,
            image_grid_size,
            text_sequence_length=text_sequence_length,
            theta=self.theta,
            use_real=True,
        )
        return image_rotary_emb, text_rotary_emb

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        encoder_hidden_states_mask: torch.Tensor = None,
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        is_multi_item = (len(hidden_states.shape) == 6)
        num_items = 0
        if is_multi_item:
            num_items = hidden_states.shape[1]
            if num_items > 1:
                assert self.patch_size[0] == 1, "For multi-item input, patch_size[0] must be 1"
                hidden_states = torch.cat(
                    [
                        hidden_states[:, -1:],
                        hidden_states[:, :-1]
                    ],
                    dim=1
                )
            batch_size, num_items, channels, frames_per_item, height, width = hidden_states.shape
            hidden_states = hidden_states.permute(0, 2, 1, 3, 4, 5).reshape(
                batch_size, channels, num_items * frames_per_item, height, width
            )

        _, _, output_frames, output_height, output_width = hidden_states.shape
        latent_frames, latent_height, latent_width = (
            output_frames // self.patch_size[0],
            output_height // self.patch_size[1],
            output_width // self.patch_size[2],
        )
        image_hidden_states = self.img_in(hidden_states).flatten(2).transpose(1, 2)

        if encoder_hidden_states_mask is None:
            encoder_hidden_states_mask = torch.ones(
                (encoder_hidden_states.shape[0], encoder_hidden_states.shape[1]),
                dtype=torch.bool,
                device=image_hidden_states.device,
            )
        else:
            encoder_hidden_states_mask = encoder_hidden_states_mask.to(device=image_hidden_states.device, dtype=torch.bool)
        modulation_states, text_hidden_states = self.condition_embedder(timestep, encoder_hidden_states)
        if modulation_states.shape[-1] > self.hidden_size:
            modulation_states = modulation_states.unflatten(1, (6, -1))

        text_seq_len = text_hidden_states.shape[1]
        image_seq_len = image_hidden_states.shape[1]
        image_rotary_emb, text_rotary_emb = self.get_rotary_pos_embed(
            image_grid_size=(latent_frames, latent_height, latent_width),
            text_sequence_length=text_seq_len if self.rope_type == 'mrope' else None,
        )

        attention_mask = torch.cat(
            [
                torch.ones(
                    (encoder_hidden_states_mask.shape[0], image_seq_len),
                    dtype=torch.bool,
                    device=encoder_hidden_states_mask.device,
                ),
                encoder_hidden_states_mask.bool(),
            ],
            dim=1,
        )
        attention_kwargs = {
            'thw': [latent_frames, latent_height, latent_width],
            'txt_len': text_seq_len,
            'attention_mask': attention_mask,
        }

        for block in self.double_blocks:
            image_hidden_states, text_hidden_states = block(
                image_hidden_states,
                text_hidden_states,
                modulation_states,
                image_rotary_emb,
                text_rotary_emb,
                attention_kwargs,
            )

        image_seq_len = image_hidden_states.shape[1]
        hidden_states = torch.cat((image_hidden_states, text_hidden_states), dim=1)
        image_hidden_states = hidden_states[:, :image_seq_len, ...]
        image_hidden_states = self.proj_out(self.norm_out(image_hidden_states))
        image_hidden_states = self.unpatchify(image_hidden_states, latent_frames, latent_height, latent_width)

        if is_multi_item:
            batch_size, channels, total_frames, height, width = image_hidden_states.shape
            image_hidden_states = image_hidden_states.reshape(
                batch_size, channels, num_items, total_frames // num_items, height, width
            ).permute(0, 2, 1, 3, 4, 5)
            if num_items > 1:
                image_hidden_states = torch.cat(
                    [
                        image_hidden_states[:, 1:],
                        image_hidden_states[:, :1],
                    ],
                    dim=1,
                )

        if return_dict:
            return {"sample": image_hidden_states, "encoder_hidden_states": text_hidden_states}
        return image_hidden_states, text_hidden_states

    def unpatchify(self, hidden_states, latent_frames, latent_height, latent_width):
        channels = self.out_channels
        patch_frames, patch_height, patch_width = self.patch_size
        assert latent_frames * latent_height * latent_width == hidden_states.shape[1]

        hidden_states = hidden_states.reshape(
            shape=(
                hidden_states.shape[0],
                latent_frames,
                latent_height,
                latent_width,
                patch_frames,
                patch_height,
                patch_width,
                channels,
            )
        )
        hidden_states = torch.einsum("nthwopqc->nctohpwq", hidden_states)

        return hidden_states.reshape(
            shape=(
                hidden_states.shape[0],
                channels,
                latent_frames * patch_frames,
                latent_height * patch_height,
                latent_width * patch_width,
            )
        )


__all__ = ["JoyAIImageTransformer3DModel"]
