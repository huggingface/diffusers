import math
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F


from ...configuration_utils import ConfigMixin, register_to_config
from ..attention import FeedForward
from ..embeddings import PixArtAlphaTextProjection, TimestepEmbedding, Timesteps
from ..modeling_outputs import Transformer2DModelOutput
from ..modeling_utils import ModelMixin

ATTN_BACKEND = 'sdpa'
try:
    from flash_attn.flash_attn_interface import flash_attn_varlen_func
    ATTN_BACKEND = 'flash_attn'
except:
    pass

def _to_tuple(x, dim=2):
    if isinstance(x, int):
        return (x,) * dim
    if len(x) == dim:
        return tuple(x)
    raise ValueError(f"Expected length {dim} or int, but got {x}")

def get_meshgrid_nd(start, *args, dim=2):
    if len(args) == 0:
        num = _to_tuple(start, dim=dim)
        start = (0,) * dim
        stop = num
    elif len(args) == 1:
        start = _to_tuple(start, dim=dim)
        stop = _to_tuple(args[0], dim=dim)
        num = [stop[i] - start[i] for i in range(dim)]
    elif len(args) == 2:
        start = _to_tuple(start, dim=dim)
        stop = _to_tuple(args[0], dim=dim)
        num = _to_tuple(args[1], dim=dim)
    else:
        raise ValueError(f"len(args) should be 0, 1 or 2, but got {len(args)}")

    axis_grid = []
    for i in range(dim):
        a, b, n = start[i], stop[i], num[i]
        g = torch.linspace(a, b, n + 1, dtype=torch.float32)[:n]
        axis_grid.append(g)
    grid = torch.meshgrid(*axis_grid, indexing="ij")
    return torch.stack(grid, dim=0)

def reshape_for_broadcast(freqs_cis, x: torch.Tensor, head_first: bool = False):
    ndim = x.ndim
    assert 0 <= 1 < ndim

    if isinstance(freqs_cis, tuple):
        if head_first:
            assert freqs_cis[0].shape == (x.shape[-2], x.shape[-1])
            shape = [d if i == ndim - 2 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        else:
            assert freqs_cis[0].shape == (x.shape[1], x.shape[-1])
            shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return freqs_cis[0].view(*shape), freqs_cis[1].view(*shape)

    if head_first:
        assert freqs_cis.shape == (x.shape[-2], x.shape[-1])
        shape = [d if i == ndim - 2 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    else:
        assert freqs_cis.shape == (x.shape[1], x.shape[-1])
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def rotate_half(x: torch.Tensor):
    x_real, x_imag = x.float().reshape(*x.shape[:-1], -1, 2).unbind(-1)
    return torch.stack([-x_imag, x_real], dim=-1).flatten(3)


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis, head_first: bool = False):
    cos, sin = reshape_for_broadcast(freqs_cis, xq, head_first)
    cos, sin = cos.to(xq.device), sin.to(xq.device)
    xq_out = (xq.float() * cos + rotate_half(xq.float()) * sin).type_as(xq)
    xk_out = (xk.float() * cos + rotate_half(xk.float()) * sin).type_as(xk)
    return xq_out, xk_out


def get_1d_rotary_pos_embed(
    dim: int,
    pos,
    theta: float = 10000.0,
    use_real: bool = False,
    theta_rescale_factor: float = 1.0,
    interpolation_factor: float = 1.0,
):
    if isinstance(pos, int):
        pos = torch.arange(pos).float()

    if theta_rescale_factor != 1.0:
        theta *= theta_rescale_factor ** (dim / (dim - 2))

    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32)[: (dim // 2)] / dim))
    freqs = torch.outer(pos.float() * interpolation_factor, freqs)

    if use_real:
        freqs_cos = freqs.cos().repeat_interleave(2, dim=1)
        freqs_sin = freqs.sin().repeat_interleave(2, dim=1)
        return freqs_cos, freqs_sin

    return torch.polar(torch.ones_like(freqs), freqs)


def get_nd_rotary_pos_embed(
    rope_dim_list,
    start,
    *args,
    theta=10000.0,
    use_real=False,
    txt_rope_size=None,
    theta_rescale_factor=1.0,
    interpolation_factor=1.0,
):
    rope_dim_list = list(rope_dim_list)
    grid = get_meshgrid_nd(start, *args, dim=len(rope_dim_list))

    if isinstance(theta_rescale_factor, (int, float)):
        theta_rescale_factor = [float(theta_rescale_factor)] * len(rope_dim_list)
    elif isinstance(theta_rescale_factor, list) and len(theta_rescale_factor) == 1:
        theta_rescale_factor = [float(theta_rescale_factor[0])] * len(rope_dim_list)

    if isinstance(interpolation_factor, (int, float)):
        interpolation_factor = [float(interpolation_factor)] * len(rope_dim_list)
    elif isinstance(interpolation_factor, list) and len(interpolation_factor) == 1:
        interpolation_factor = [float(interpolation_factor[0])] * len(rope_dim_list)

    embs = []
    for i in range(len(rope_dim_list)):
        emb = get_1d_rotary_pos_embed(
            rope_dim_list[i],
            grid[i].reshape(-1),
            theta,
            use_real=use_real,
            theta_rescale_factor=theta_rescale_factor[i],
            interpolation_factor=interpolation_factor[i],
        )
        embs.append(emb)

    if use_real:
        vis_emb = (torch.cat([emb[0] for emb in embs], dim=1), torch.cat([emb[1] for emb in embs], dim=1))
    else:
        vis_emb = torch.cat(embs, dim=1)

    if txt_rope_size is None:
        return vis_emb, None

    embs_txt = []
    vis_max_ids = grid.view(-1).max().item()
    grid_txt = torch.arange(txt_rope_size) + vis_max_ids + 1
    for i in range(len(rope_dim_list)):
        emb = get_1d_rotary_pos_embed(
            rope_dim_list[i],
            grid_txt,
            theta,
            use_real=use_real,
            theta_rescale_factor=theta_rescale_factor[i],
            interpolation_factor=interpolation_factor[i],
        )
        embs_txt.append(emb)

    if use_real:
        txt_emb = (torch.cat([emb[0] for emb in embs_txt], dim=1), torch.cat([emb[1] for emb in embs_txt], dim=1))
    else:
        txt_emb = torch.cat(embs_txt, dim=1)

    return vis_emb, txt_emb

def get_cu_seqlens(text_mask, img_len):
    """Calculate cu_seqlens_q, cu_seqlens_kv using text_mask and img_len

    Args:
        text_mask (torch.Tensor): the mask of text
        img_len (int): the length of image

    Returns:
        torch.Tensor: the calculated cu_seqlens for flash attention
    """
    batch_size = text_mask.shape[0]
    text_len = text_mask.sum(dim=1)
    max_len = text_mask.shape[1] + img_len

    cu_seqlens = torch.zeros([2 * batch_size + 1],
                             dtype=torch.int32, device="cuda")

    for i in range(batch_size):
        s = text_len[i] + img_len
        s1 = i * max_len + s
        s2 = (i + 1) * max_len
        cu_seqlens[2 * i + 1] = s1
        cu_seqlens[2 * i + 2] = s2

    return cu_seqlens

def load_modulation(modulate_type: str, hidden_size: int, factor: int, act_layer=nn.SiLU, dtype=None, device=None):
    factory_kwargs = {"dtype": dtype, "device": device}
    if modulate_type == "wanx":
        return ModulateWan(hidden_size, factor, **factory_kwargs)
    if modulate_type == "adaLN":
        return ModulateDiT(hidden_size, factor, act_layer, **factory_kwargs)
    if modulate_type == "jdx":
        return ModulateX(hidden_size, factor, **factory_kwargs)
    raise ValueError(f"Unknown modulation type: {modulate_type}.")


class ModulateDiT(nn.Module):
    def __init__(self, hidden_size: int, factor: int, act_layer=nn.SiLU, dtype=None, device=None):
        factory_kwargs = {"dtype": dtype, "device": device}
        super().__init__()
        self.factor = factor
        self.act = act_layer()
        self.linear = nn.Linear(hidden_size, factor * hidden_size, bias=True, **factory_kwargs)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor):
        return self.linear(self.act(x)).chunk(self.factor, dim=-1)


class ModulateWan(nn.Module):
    def __init__(self, hidden_size: int, factor: int, dtype=None, device=None):
        super().__init__()
        self.factor = factor
        self.modulate_table = nn.Parameter(
            torch.zeros(1, factor, hidden_size, dtype=dtype, device=device) / hidden_size**0.5, requires_grad=True
        )

    def forward(self, x: torch.Tensor):
        if len(x.shape) != 3:
            x = x.unsqueeze(1)
        return [o.squeeze(1) for o in (self.modulate_table + x).chunk(self.factor, dim=1)]


class ModulateX(nn.Module):
    def __init__(self, hidden_size: int, factor: int, dtype=None, device=None):
        super().__init__()
        self.factor = factor

    def forward(self, x: torch.Tensor):
        if len(x.shape) != 3:
            x = x.unsqueeze(1)
        return [o.squeeze(1) for o in x.chunk(self.factor, dim=1)]


def modulate(x, shift=None, scale=None):
    if scale is None and shift is None:
        return x
    if shift is None:
        return x * (1 + scale.unsqueeze(1))
    if scale is None:
        return x + shift.unsqueeze(1)
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def apply_gate(x, gate=None, tanh=False):
    if gate is None:
        return x
    return x * (gate.unsqueeze(1).tanh() if tanh else gate.unsqueeze(1))

def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_kwargs=None):
    batch_size = q.shape[0]
    if ATTN_BACKEND == 'sdpa':
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_kwargs['attn_mask'])
        output = output.transpose(1, 2)
    elif ATTN_BACKEND == 'flash_attn':
        cu_seqlens_q = attn_kwargs['cu_seqlens_q']
        cu_seqlens_kv = attn_kwargs['cu_seqlens_kv']
        max_seqlen_q = attn_kwargs['max_seqlen_q']
        max_seqlen_kv = attn_kwargs['max_seqlen_kv']
        x = flash_attn_varlen_func(
            q.view(q.shape[0] * q.shape[1], *q.shape[2:]),
            k.view(k.shape[0] * k.shape[1], *k.shape[2:]),
            v.view(v.shape[0] * v.shape[1], *v.shape[2:]),
            cu_seqlens_q,
            cu_seqlens_kv,
            max_seqlen_q,
            max_seqlen_kv,
        )
        # x with shape [(bxs), a, d]
        output = x.view(
            batch_size, max_seqlen_q, x.shape[-2], x.shape[-1]
        )  # reshape
    return output

class RMSNorm(nn.Module):
    def __init__(self, dim: int, elementwise_affine=True, eps: float = 1e-6, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.eps = eps
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim, **factory_kwargs))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        if hasattr(self, "weight"):
            output = output * self.weight
        return output


class MMDoubleStreamBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        heads_num: int,
        mlp_width_ratio: float,
        dtype=None,
        device=None,
        dit_modulation_type: str = "wanx",
        attn_backend: str = "torch_spda",
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.attn_backend = attn_backend
        self.dit_modulation_type = dit_modulation_type
        self.heads_num = heads_num
        head_dim = hidden_size // heads_num
        mlp_hidden_dim = int(hidden_size * mlp_width_ratio)

        self.img_mod = load_modulation(self.dit_modulation_type, hidden_size, 6, **factory_kwargs)
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)
        self.img_attn_qkv = nn.Linear(hidden_size, hidden_size * 3, bias=True, **factory_kwargs)
        self.img_attn_q_norm = RMSNorm(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
        self.img_attn_k_norm = RMSNorm(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
        self.img_attn_proj = nn.Linear(hidden_size, hidden_size, bias=True, **factory_kwargs)
        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)
        self.img_mlp = FeedForward(hidden_size, inner_dim=mlp_hidden_dim, activation_fn="gelu-approximate")

        self.txt_mod = load_modulation(self.dit_modulation_type, hidden_size, 6, **factory_kwargs)
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)
        self.txt_attn_qkv = nn.Linear(hidden_size, hidden_size * 3, bias=True, **factory_kwargs)
        self.txt_attn_q_norm = RMSNorm(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
        self.txt_attn_k_norm = RMSNorm(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
        self.txt_attn_proj = nn.Linear(hidden_size, hidden_size, bias=True, **factory_kwargs)
        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)
        self.txt_mlp = FeedForward(hidden_size, inner_dim=mlp_hidden_dim, activation_fn="gelu-approximate")

    def forward(self, img: torch.Tensor, txt: torch.Tensor, vec: torch.Tensor, vis_freqs_cis=None, txt_freqs_cis=None, attn_kwargs=None):
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

        img_modulated = modulate(self.img_norm1(img), shift=img_mod1_shift, scale=img_mod1_scale)
        img_qkv = self.img_attn_qkv(img_modulated)
        B, L, _ = img_qkv.shape
        img_q, img_k, img_v = img_qkv.reshape(B, L, 3, self.heads_num, -1).permute(2, 0, 1, 3, 4).unbind(0)
        img_q = self.img_attn_q_norm(img_q).to(img_v)
        img_k = self.img_attn_k_norm(img_k).to(img_v)
        if vis_freqs_cis is not None:
            img_q, img_k = apply_rotary_emb(img_q, img_k, vis_freqs_cis, head_first=False)

        txt_modulated = modulate(self.txt_norm1(txt), shift=txt_mod1_shift, scale=txt_mod1_scale)
        txt_qkv = self.txt_attn_qkv(txt_modulated)
        B2, L2, _ = txt_qkv.shape
        txt_q, txt_k, txt_v = txt_qkv.reshape(B2, L2, 3, self.heads_num, -1).permute(2, 0, 1, 3, 4).unbind(0)
        txt_q = self.txt_attn_q_norm(txt_q).to(txt_v)
        txt_k = self.txt_attn_k_norm(txt_k).to(txt_v)
        if txt_freqs_cis is not None:
            txt_q, txt_k = apply_rotary_emb(txt_q, txt_k, txt_freqs_cis, head_first=False)

        q = torch.cat((img_q, txt_q), dim=1)
        k = torch.cat((img_k, txt_k), dim=1)
        v = torch.cat((img_v, txt_v), dim=1)

        attn = attention(q, k, v, attn_kwargs=attn_kwargs).flatten(2, 3)
        img_attn, txt_attn = attn[:, : img.shape[1]], attn[:, img.shape[1] :]

        img = img + apply_gate(self.img_attn_proj(img_attn), gate=img_mod1_gate)
        img = img + apply_gate(
            self.img_mlp(modulate(self.img_norm2(img), shift=img_mod2_shift, scale=img_mod2_scale)),
            gate=img_mod2_gate,
        )

        txt = txt + apply_gate(self.txt_attn_proj(txt_attn), gate=txt_mod1_gate)
        txt = txt + apply_gate(
            self.txt_mlp(modulate(self.txt_norm2(txt), shift=txt_mod2_shift, scale=txt_mod2_scale)),
            gate=txt_mod2_gate,
        )

        return img, txt


class WanTimeTextImageEmbedding(nn.Module):
    def __init__(self, dim: int, time_freq_dim: int, time_proj_dim: int, text_embed_dim: int):
        super().__init__()
        self.timesteps_proj = Timesteps(num_channels=time_freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedder = TimestepEmbedding(in_channels=time_freq_dim, time_embed_dim=dim)
        self.act_fn = nn.SiLU()
        self.time_proj = nn.Linear(dim, time_proj_dim)
        self.text_embedder = PixArtAlphaTextProjection(text_embed_dim, dim, act_fn="gelu_tanh")

    def forward(self, timestep: torch.Tensor, encoder_hidden_states: torch.Tensor):
        timestep = self.timesteps_proj(timestep)
        time_embedder_dtype = next(iter(self.time_embedder.parameters())).dtype
        if timestep.dtype != time_embedder_dtype and time_embedder_dtype != torch.int8:
            timestep = timestep.to(time_embedder_dtype)
        temb = self.time_embedder(timestep).type_as(encoder_hidden_states)
        timestep_proj = self.time_proj(self.act_fn(temb))
        encoder_hidden_states = self.text_embedder(encoder_hidden_states)
        return temb, timestep_proj, encoder_hidden_states


class JoyImageTransformer3DModel(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        patch_size: tuple[int, int, int] = (1, 2, 2),
        in_channels: int = 16,
        out_channels: int = 16,
        hidden_size: int = 4096,
        heads_num: int = 32,
        text_states_dim: int = 4096,
        mlp_width_ratio: float = 4.0,
        mm_double_blocks_depth: int = 40,
        rope_dim_list: tuple[int, int, int] = (16, 56, 56),
        rope_type: str = "rope",
        dit_modulation_type: str = "wanx",
        attn_backend: str = "torch_spda",
        unpatchify_new: bool = True,
        rope_theta: int = 256,
        enable_activation_checkpointing: bool = False,
        is_repa: bool = False,
        repa_layer: int = 13,
    ):
        super().__init__()

        self.args = SimpleNamespace(
            enable_activation_checkpointing=enable_activation_checkpointing,
            is_repa=is_repa,
            repa_layer=repa_layer,
        )

        self.out_channels = out_channels or in_channels
        self.patch_size = tuple(patch_size)
        self.hidden_size = hidden_size
        self.heads_num = heads_num
        self.rope_dim_list = tuple(rope_dim_list)
        self.dit_modulation_type = dit_modulation_type
        self.mm_double_blocks_depth = mm_double_blocks_depth
        self.attn_backend = attn_backend
        self.rope_type = rope_type
        self.unpatchify_new = unpatchify_new
        self.theta = rope_theta

        if hidden_size % heads_num != 0:
            raise ValueError(f"Hidden size {hidden_size} must be divisible by heads_num {heads_num}")

        self.img_in = nn.Conv3d(in_channels, hidden_size, kernel_size=self.patch_size, stride=self.patch_size)

        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=hidden_size,
            time_freq_dim=256,
            time_proj_dim=hidden_size * 6 if dit_modulation_type != "adaLN" else hidden_size,
            text_embed_dim=text_states_dim,
        )

        self.double_blocks = nn.ModuleList(
            [
                MMDoubleStreamBlock(
                    hidden_size=self.hidden_size,
                    heads_num=self.heads_num,
                    mlp_width_ratio=mlp_width_ratio,
                    dit_modulation_type=self.dit_modulation_type,
                    attn_backend=attn_backend,
                )
                for _ in range(mm_double_blocks_depth)
            ]
        )

        self.norm_out = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(hidden_size, out_channels * math.prod(self.patch_size))

        if self.args.is_repa:
            self.repa_proj = nn.Linear(hidden_size, text_states_dim)
            if self.args.repa_layer > mm_double_blocks_depth:
                raise ValueError("repa_layer should be smaller than total depth")

        self.gradient_checkpointing = enable_activation_checkpointing

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def get_rotary_pos_embed(self, vis_rope_size, txt_rope_size=None):
        target_ndim = 3
        if len(vis_rope_size) != target_ndim:
            vis_rope_size = [1] * (target_ndim - len(vis_rope_size)) + list(vis_rope_size)

        head_dim = self.hidden_size // self.heads_num
        rope_dim_list = list(self.rope_dim_list)
        if rope_dim_list is None:
            rope_dim_list = [head_dim // target_ndim for _ in range(target_ndim)]
        if sum(rope_dim_list) != head_dim:
            raise ValueError("sum(rope_dim_list) should equal head_dim")

        return get_nd_rotary_pos_embed(
            rope_dim_list,
            vis_rope_size,
            txt_rope_size=txt_rope_size,
            theta=self.theta,
            use_real=True,
            theta_rescale_factor=1,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        encoder_hidden_states_mask: torch.Tensor = None,
        return_dict: bool = True,
    ):
        if encoder_hidden_states is None:
            raise ValueError("encoder_hidden_states is required.")

        is_multi_item = len(hidden_states.shape) == 6
        num_items = 0
        if is_multi_item:
            num_items = hidden_states.shape[1]
            if num_items > 1:
                if self.patch_size[0] != 1:
                    raise ValueError("For multi-item input, patch_size[0] must be 1")
                hidden_states = torch.cat([hidden_states[:, -1:], hidden_states[:, :-1]], dim=1)
            hidden_states = hidden_states.permute(0, 2, 1, 3, 4, 5).flatten(2, 3)

        _, _, ot, oh, ow = hidden_states.shape
        tt, th, tw = (
            ot // self.patch_size[0],
            oh // self.patch_size[1],
            ow // self.patch_size[2],
        )

        if encoder_hidden_states_mask is None:
            encoder_hidden_states_mask = torch.ones(
                (encoder_hidden_states.shape[0], encoder_hidden_states.shape[1]),
                dtype=torch.bool,
                device=encoder_hidden_states.device,
            )

        img = self.img_in(hidden_states).flatten(2).transpose(1, 2)
        _, vec, txt = self.condition_embedder(timestep, encoder_hidden_states)
        if vec.shape[-1] > self.hidden_size:
            vec = vec.unflatten(1, (6, -1))

        txt_seq_len = txt.shape[1]
        vis_freqs_cis, txt_freqs_cis = self.get_rotary_pos_embed(
            vis_rope_size=(tt, th, tw),
            txt_rope_size=txt_seq_len if self.rope_type == "mrope" else None,
        )

        txt_seq_len = txt.shape[1]
        img_seq_len = img.shape[1]
        
        cu_seqlens_q = get_cu_seqlens(
                encoder_hidden_states_mask, img_seq_len)
        cu_seqlens_kv = cu_seqlens_q
        max_seqlen_q = img_seq_len + txt_seq_len
        max_seqlen_kv = max_seqlen_q


        attn_kwargs = {"encoder_hidden_states_mask": encoder_hidden_states_mask}
        attn_kwargs.update({
                'cu_seqlens_q': cu_seqlens_q,
                'cu_seqlens_kv': cu_seqlens_kv,
                'max_seqlen_q': max_seqlen_q,
                'max_seqlen_kv': max_seqlen_kv,
            })
    
        max_seqlen_q = img_seq_len + txt_seq_len
        seq_lens = encoder_hidden_states_mask.sum(dim=1) + img_seq_len
        max_len = encoder_hidden_states_mask.shape[1] + img_seq_len
        assert max_seqlen_q == max_len
        positions = torch.arange(max_seqlen_q, device=img.device).unsqueeze(0)
        seq_lens_expanded = seq_lens.unsqueeze(1)
        mask = positions < seq_lens_expanded
        mask = mask.unsqueeze(1).unsqueeze(2)
        attn_mask = mask & mask.transpose(-1, -2)

        attn_kwargs.update({'attn_mask': attn_mask, 'max_seqlen_q': max_seqlen_q})
    
        img_hidden_states = []
        for block in self.double_blocks:
            img, txt = block(img, txt, vec, vis_freqs_cis, txt_freqs_cis, attn_kwargs)
            img_hidden_states.append(img)

        img_len = img.shape[1]
        x = torch.cat((img, txt), 1)
        img = x[:, :img_len, ...]

        img = self.proj_out(self.norm_out(img))
        img = self.unpatchify(img, tt, th, tw)

        repa_hidden_state = None
        if self.args.is_repa:
            repa_hidden_state = self.repa_proj(img_hidden_states[self.args.repa_layer])
            repa_hidden_state = repa_hidden_state.view(img.shape[0], tt, th, tw, -1)

        if is_multi_item:
            # b c (n t) h w -> b n c t h w
            b, c, nt, h, w = img.shape
            t = nt // num_items
            img = img.reshape(b, c, num_items, t, h, w).permute(0, 2, 1, 3, 4, 5)
            if num_items > 1:
                img = torch.cat([img[:, 1:], img[:, :1]], dim=1)
            if repa_hidden_state is not None:
                # b (n t) h w c -> b n t h w c
                b2, nt2, h2, w2, c2 = repa_hidden_state.shape
                t2 = nt2 // num_items
                repa_hidden_state = repa_hidden_state.reshape(b2, num_items, t2, h2, w2, c2)

        if not return_dict:
            return (img, txt, repa_hidden_state)

        return Transformer2DModelOutput(sample=img)

    def unpatchify(self, x: torch.Tensor, t: int, h: int, w: int):
        c = self.out_channels
        pt, ph, pw = self.patch_size
        if t * h * w != x.shape[1]:
            raise ValueError("Invalid token length for unpatchify.")

        if self.unpatchify_new:
            x = x.reshape(shape=(x.shape[0], t, h, w, pt, ph, pw, c))
            x = torch.einsum("nthwopqc->nctohpwq", x)
        else:
            x = x.reshape(shape=(x.shape[0], t, h, w, c, pt, ph, pw))
            x = torch.einsum("nthwcopq->nctohpwq", x)

        return x.reshape(shape=(x.shape[0], c, t * pt, h * ph, w * pw))


class JoyImageEditTransformer3DModel(JoyImageTransformer3DModel):
    """
    Backward-compatible alias of JoyImageTransformer3DModel.
    """

    pass