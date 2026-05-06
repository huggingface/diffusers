# Copyright 2025 The CogView team, Tsinghua University & ZhipuAI and The HuggingFace Team. All rights reserved.
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

"""
Image tokenizer for UniLLaDA.
Converts PIL images into discrete VQ token IDs via a vision encoder + VQVAE.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import PIL.Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.v2 import functional as tvF


# ============================================================
# Config loading
# ============================================================


def load_configs(model_dir: str | Path) -> dict:
    with open(Path(model_dir) / "config.json", "r") as f:
        return json.load(f)


def make_vision_config(raw: dict) -> SimpleNamespace:
    vc = raw.get("vision_config", raw)
    # Determine best attention implementation
    attn_impl = "eager"
    try:
        from flash_attn import flash_attn_varlen_func  # noqa: F401

        attn_impl = "flash_attention_2"
    except ImportError:
        try:
            import torch

            if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
                attn_impl = "sdpa"
        except Exception:
            pass
    return SimpleNamespace(
        hidden_size=vc["hidden_size"],
        intermediate_size=vc["intermediate_size"],
        num_heads=vc["num_heads"],
        depth=vc["depth"],
        patch_size=vc["patch_size"],
        image_size=vc["image_size"],
        in_channels=vc.get("in_channels", 3),
        hidden_act=vc.get("hidden_act", "gelu"),
        attention_bias=vc.get("attention_bias", True),
        attention_dropout=vc.get("attention_dropout", 0.0),
        layer_norm_eps=vc.get("layer_norm_eps", 1e-6),
        spatial_merge_size=vc.get("spatial_merge_size", 1),
        _attn_implementation=attn_impl,
    )


def make_vq_config(raw: dict) -> SimpleNamespace:
    vq = raw.get("vq_config", raw)
    return SimpleNamespace(
        num_embeddings=vq["num_embeddings"],
        embed_dim=vq["embed_dim"],
        latent_channels=vq["latent_channels"],
        beta=vq.get("beta", 0.25),
    )


# ============================================================
# Image preprocessing
# ============================================================

OPENAI_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


class ImagePreprocessor:
    """Image preprocessor: rescale + normalize. Resizing/cropping is handled externally."""

    def __init__(self, config_path: str | Path):
        config_path = Path(config_path)
        if config_path.is_dir():
            config_path = config_path / "preprocessor_config.json"
        with open(config_path, "r") as f:
            config = json.load(f)

        self.do_rescale = config.get("do_rescale", True)
        self.do_normalize = config.get("do_normalize", True)
        self.rescale_factor = config.get("rescale_factor", 1.0 / 255.0)
        self.image_mean = config.get("image_mean", OPENAI_CLIP_MEAN)
        self.image_std = config.get("image_std", OPENAI_CLIP_STD)
        self.patch_size = config.get("patch_size", 14)
        self.temporal_patch_size = config.get("temporal_patch_size", 2)
        self.merge_size = config.get("merge_size", 2)
        self.factor = self.patch_size * self.merge_size

    def _pil_to_tensor(self, image):
        return tvF.to_dtype(tvF.to_image(image), dtype=torch.float32, scale=False)

    def _rescale_and_normalize(self, images):
        if self.do_rescale:
            images = images * self.rescale_factor
        if self.do_normalize:
            mean = torch.tensor(self.image_mean, dtype=images.dtype, device=images.device).view(-1, 1, 1)
            std = torch.tensor(self.image_std, dtype=images.dtype, device=images.device).view(-1, 1, 1)
            images = (images - mean) / std
        return images

    def __call__(self, images):
        if isinstance(images, PIL.Image.Image):
            images = [images]

        all_patches, all_grids = [], []
        for img in images:
            if img.mode != "RGB":
                img = img.convert("RGB")
            img_tensor = self._pil_to_tensor(img)
            height, width = img_tensor.shape[-2:]
            rh, rw = height, width

            patches = self._rescale_and_normalize(img_tensor)
            if patches.ndim == 3:
                patches = patches.unsqueeze(0)

            # Temporal padding
            if patches.shape[0] % self.temporal_patch_size != 0:
                repeats = patches[-1:].repeat(self.temporal_patch_size - 1, 1, 1, 1)
                patches = torch.cat([patches, repeats], dim=0)

            grid_t = patches.shape[0] // self.temporal_patch_size
            grid_h = rh // self.patch_size
            grid_w = rw // self.patch_size
            channel = patches.shape[1]

            # Reshape into patch tokens
            patches = patches.unsqueeze(0).view(
                1,
                grid_t,
                self.temporal_patch_size,
                channel,
                grid_h // self.merge_size,
                self.merge_size,
                self.patch_size,
                grid_w // self.merge_size,
                self.merge_size,
                self.patch_size,
            )
            patches = patches.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
            flatten_patches = patches.reshape(
                1,
                grid_t * grid_h * grid_w,
                channel * self.temporal_patch_size * self.patch_size * self.patch_size,
            )
            all_patches.append(flatten_patches.squeeze(0))
            all_grids.append([grid_t, grid_h, grid_w])

        return {
            "pixel_values": torch.cat(all_patches, dim=0),
            "image_grid_thw": torch.tensor(all_grids, dtype=torch.long),
        }


# ============================================================
# Vision model components
# ============================================================


def _get_act_fn(name):
    mapping = {
        "gelu": nn.GELU(),
        "relu": nn.ReLU(),
        "silu": nn.SiLU(),
        "quick_gelu": lambda x: x * torch.sigmoid(1.702 * x),
    }
    if name in mapping:
        return mapping[name]
    from transformers.activations import ACT2FN

    return ACT2FN[name]


class VisionMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.activation_fn = _get_act_fn(config.hidden_act)
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, x):
        return self.fc2(self.activation_fn(self.fc1(x)))


class VisionAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.dim // self.num_heads
        self.num_key_value_groups = 1
        self.qkv = nn.Linear(config.hidden_size, config.hidden_size * 3, bias=config.attention_bias)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)
        self.scaling = self.head_dim**-0.5
        self.config = config
        self.attention_dropout = config.attention_dropout
        self.is_causal = False

    def forward(self, hidden_states, cu_seqlens, **kwargs):
        seq_length = hidden_states.shape[0]
        query_states, key_states, value_states = (
            self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        )
        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        # Try to use the HF attention dispatch (flash_attention_2 / sdpa / eager)
        try:
            from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

            attn_impl = getattr(self.config, "_attn_implementation", "eager")
            if attn_impl != "eager" and attn_impl in ALL_ATTENTION_FUNCTIONS:
                attention_interface = ALL_ATTENTION_FUNCTIONS[attn_impl]
                if "flash" in attn_impl:
                    max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
                    attn_output, _ = attention_interface(
                        self,
                        query_states,
                        key_states,
                        value_states,
                        attention_mask=None,
                        scaling=self.scaling,
                        dropout=0.0 if not self.training else self.attention_dropout,
                        cu_seq_lens_q=cu_seqlens,
                        cu_seq_lens_k=cu_seqlens,
                        max_length_q=max_seqlen,
                        max_length_k=max_seqlen,
                        is_causal=False,
                        **kwargs,
                    )
                else:
                    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
                    splits = [
                        torch.split(t, lengths.tolist(), dim=2) for t in (query_states, key_states, value_states)
                    ]
                    attn_output = torch.cat(
                        [
                            attention_interface(
                                self,
                                q,
                                k,
                                v,
                                attention_mask=None,
                                scaling=self.scaling,
                                dropout=0.0 if not self.training else self.attention_dropout,
                                is_causal=False,
                                **kwargs,
                            )[0]
                            for q, k, v in zip(*splits)
                        ],
                        dim=1,
                    )
                attn_output = attn_output.reshape(seq_length, -1).contiguous()
                return self.proj(attn_output)
        except (ImportError, KeyError, AttributeError):
            pass

        # Fallback: try flash_attn directly
        try:
            from flash_attn import flash_attn_varlen_func

            q = query_states.squeeze(0).transpose(0, 1)
            k = key_states.squeeze(0).transpose(0, 1)
            v = value_states.squeeze(0).transpose(0, 1)
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
            attn_output = flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen)
            attn_output = attn_output.reshape(seq_length, -1).contiguous()
            return self.proj(attn_output)
        except ImportError:
            pass

        # Final fallback: manual eager attention
        q = query_states.squeeze(0)
        k = key_states.squeeze(0)
        v = value_states.squeeze(0)
        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        outputs = []
        for qc, kc, vc in zip(
            torch.split(q, lengths, dim=1), torch.split(k, lengths, dim=1), torch.split(v, lengths, dim=1)
        ):
            attn = F.softmax(torch.matmul(qc, kc.transpose(-2, -1)) * self.scaling, dim=-1, dtype=torch.float32).to(
                qc.dtype
            )
            outputs.append(torch.matmul(attn, vc))
        attn_output = torch.cat(outputs, dim=1).transpose(0, 1).reshape(seq_length, -1).contiguous()
        return self.proj(attn_output)


class VisionPatchEmbed(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.patch_size = config.patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.hidden_size
        self.proj = nn.Conv2d(self.in_channels, self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, x):
        target_dtype = self.proj.weight.dtype
        x = x.view(-1, self.in_channels, self.patch_size, self.patch_size)
        return self.proj(x.to(dtype=target_dtype)).view(-1, self.embed_dim)


class VisionEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.hidden_size
        num_patches = (config.image_size // config.patch_size) ** 2
        self.position_embedding = nn.Embedding(num_patches, self.embed_dim)

    def forward(self, embeddings, lengths, image_shapes, h_coords, w_coords):
        pos_w = self.position_embedding.weight
        hidden_size = pos_w.shape[1]
        device = pos_w.device

        if isinstance(lengths, list):
            lengths = torch.tensor(lengths, device=device, dtype=torch.long)

        orig_size = int(pos_w.shape[0] ** 0.5)
        pos_2d = pos_w.view(orig_size, orig_size, hidden_size).permute(2, 0, 1).unsqueeze(0).float()

        target_h = torch.cat([image_shapes[i, 1].repeat(lengths[i]) for i in range(len(lengths))]).to(
            device=device, dtype=torch.float32
        )
        target_w = torch.cat([image_shapes[i, 2].repeat(lengths[i]) for i in range(len(lengths))]).to(
            device=device, dtype=torch.float32
        )

        norm_w = ((w_coords + 0.5) / target_w) * 2 - 1
        norm_h = ((h_coords + 0.5) / target_h) * 2 - 1
        grid = torch.stack((norm_w, norm_h), dim=-1).unsqueeze(0).unsqueeze(2)

        adapted = F.grid_sample(pos_2d, grid, mode="bilinear", align_corners=False, padding_mode="border")
        adapted = adapted.squeeze(0).squeeze(-1).permute(1, 0).to(pos_w.dtype).to(embeddings.device)
        return embeddings + adapted


class VisionBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attn = VisionAttention(config)
        self.mlp = VisionMLP(config)

    def forward(self, hidden_states, cu_seqlens, **kwargs):
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states), cu_seqlens=cu_seqlens)
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class VisionEncoder(nn.Module):
    """Vision transformer encoder that produces per-patch features."""

    def __init__(self, config):
        super().__init__()
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        self.embeddings = VisionEmbeddings(config)
        self.patch_embed = VisionPatchEmbed(config)
        self.blocks = nn.ModuleList([VisionBlock(config) for _ in range(config.depth)])

    @property
    def dtype(self):
        return self.patch_embed.proj.weight.dtype

    def rot_pos_emb(self, grid_thw):
        pos_ids = []
        for t, h, w in grid_thw:
            hpos = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos = hpos.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos = hpos.permute(0, 2, 1, 3).flatten()

            wpos = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos = wpos.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos = wpos.permute(0, 2, 1, 3).flatten()

            pos_ids.append(torch.stack([hpos, wpos], dim=-1).repeat(t, 1))
        return torch.cat(pos_ids, dim=0)

    def forward(self, pixel_values, grid_thw):
        hidden_states = self.patch_embed(pixel_values)
        image_type_ids = self.rot_pos_emb(grid_thw)

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0])
        cu_seqlens = F.pad(cu_seqlens.cumsum(0, dtype=torch.int32), (1, 0), value=0)
        seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()

        hidden_states = self.embeddings(
            hidden_states,
            seqlens,
            grid_thw,
            image_type_ids[:, 0].to(hidden_states.device),
            image_type_ids[:, 1].to(hidden_states.device),
        )
        for blk in self.blocks:
            hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens)
        return hidden_states


# ============================================================
# VQVAE quantizer
# ============================================================


class VQVAEVectorQuantizer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_embeddings = config.num_embeddings
        self.embedding_dim = config.embed_dim
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)

    def forward(self, hidden_state):
        hidden_state = hidden_state.permute(0, 2, 3, 1).contiguous()
        flat = hidden_state.view(-1, self.embedding_dim)

        flat = F.normalize(flat, p=2, dim=-1)
        emb = F.normalize(self.embedding.weight, p=2, dim=-1)

        distances = (
            torch.sum(flat**2, dim=1, keepdim=True)
            + torch.sum(emb**2, dim=1)
            - 2 * torch.einsum("bd,dn->bn", flat, emb.t())
        )
        return torch.argmin(distances, dim=1)


class VQVAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.quantize = VQVAEVectorQuantizer(config)
        self.quant_conv = nn.Conv2d(config.latent_channels, config.embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(config.embed_dim, config.latent_channels, 1)

    def encode(self, hidden_states):
        return self.quantize(self.quant_conv(hidden_states))


# ============================================================
# Weight loading
# ============================================================


def _load_weights(model_dir, visual, vqmodel):
    from safetensors.torch import load_file

    model_path = Path(model_dir)
    index_file = model_path / "model.safetensors.index.json"

    if index_file.exists():
        with open(index_file) as f:
            weight_map = json.load(f)["weight_map"]
        needed = {fn for k, fn in weight_map.items() if k.startswith(("model.visual.", "model.vqmodel."))}
    else:
        needed = {f.name for f in model_path.glob("*.safetensors")}

    visual_sd, vq_sd = {}, {}
    for filename in sorted(needed):
        filepath = model_path / filename
        if not filepath.exists():
            continue
        shard = load_file(str(filepath), device="cpu")
        for key, value in shard.items():
            if key.startswith("model.visual."):
                visual_sd[key[len("model.visual.") :]] = value
            elif key.startswith("model.vqmodel."):
                vq_sd[key[len("model.vqmodel.") :]] = value
        del shard

    visual.load_state_dict(visual_sd, strict=False)
    vqmodel.load_state_dict(vq_sd, strict=False)
    del visual_sd, vq_sd


# ============================================================
# Main tokenizer class
# ============================================================


class ImageTokenizer:
    """
    Standalone image tokenizer that converts PIL images to discrete VQ token IDs.

    Expects the following layout under ``model_path``::

        model_path/
        └── image_tokenizer/
            ├── config.json              # vision_config + vq_config
            ├── preprocessor_config.json
            └── *.safetensors            # visual + vqmodel weights

    Args:
        model_path: Root path of the model directory (parent of image_tokenizer/).
        device: Torch device.
        dtype: Model dtype (default: bfloat16).
    """

    def __init__(self, model_path, device="cuda", dtype=torch.bfloat16):
        self.device = torch.device(device)
        self.dtype = dtype

        tokenizer_dir = Path(model_path) / "image_tokenizer"

        self.image_processor = ImagePreprocessor(tokenizer_dir)

        raw_config = load_configs(tokenizer_dir)
        vision_cfg = make_vision_config(raw_config)
        vq_cfg = make_vq_config(raw_config)

        self.visual = VisionEncoder(vision_cfg).to(self.device, self.dtype)
        self.vqmodel = VQVAE(vq_cfg).to(self.device, self.dtype)

        _load_weights(str(tokenizer_dir), self.visual, self.vqmodel)
        self.visual.eval()
        self.vqmodel.eval()
        self.spatial_merge_size = vision_cfg.spatial_merge_size

    @staticmethod
    def _whiten_transparency(img):
        if img.mode == "RGBA":
            canvas = PIL.Image.new("RGBA", img.size, (255, 255, 255, 255))
            canvas.alpha_composite(img)
            return canvas.convert("RGB")
        return img if img.mode == "RGB" else img.convert("RGB")

    def _extract_features(self, pixel_values, image_grid_thw):
        with torch.no_grad():
            hidden = self.visual(pixel_values.to(self.device, self.dtype), grid_thw=image_grid_thw.to(self.device))
        split_sizes = (image_grid_thw.prod(-1) // self.spatial_merge_size**2).tolist()
        return list(torch.split(hidden, split_sizes))

    def _quantize(self, hidden_states, image_grid_thw):
        hidden_size = hidden_states.shape[-1]
        split_sizes = image_grid_thw.prod(dim=-1).tolist()
        all_tokens = []
        with torch.no_grad():
            for i, hs in enumerate(torch.split(hidden_states, split_sizes)):
                gt, gh, gw = image_grid_thw[i].tolist()
                hs = hs.view(gt, gh, gw, hidden_size).permute(0, 3, 1, 2).contiguous()
                all_tokens.append(self.vqmodel.encode(hs))
        return torch.cat(all_tokens, dim=0)

    @torch.no_grad()
    def encode(self, image: PIL.Image.Image) -> list[int]:
        """Encode a single image to VQ token IDs."""
        image = self._whiten_transparency(image)
        inputs = self.image_processor([image])
        embeds = self._extract_features(inputs["pixel_values"], inputs["image_grid_thw"])
        tokens = self._quantize(torch.cat(embeds, dim=0), inputs["image_grid_thw"])
        return tokens.flatten().tolist()

    @torch.no_grad()
    def encode_batch(self, images: list[PIL.Image.Image]) -> list[list[int]]:
        """Encode a batch of images to VQ token IDs."""
        images = [self._whiten_transparency(img) for img in images]
        inputs = self.image_processor(images)
        pv, grid = inputs["pixel_values"], inputs["image_grid_thw"]
        embeds = self._extract_features(pv, grid)
        return [self._quantize(e, grid[i : i + 1]).flatten().tolist() for i, e in enumerate(embeds)]

    @torch.no_grad()
    def encode_with_info(self, image: PIL.Image.Image) -> dict:
        """Encode image and return token IDs with metadata."""
        image = self._whiten_transparency(image)
        w, h = image.size
        inputs = self.image_processor([image])
        pv, grid = inputs["pixel_values"], inputs["image_grid_thw"]
        embeds = self._extract_features(pv, grid)
        tl = self._quantize(torch.cat(embeds, dim=0), grid).flatten().tolist()
        return {
            "pixel_values": pv,
            "token_ids": tl,
            "grid_thw": tuple(grid[0].tolist()),
            "num_tokens": len(tl),
            "image_size": (w, h),
        }

    @property
    def codebook_size(self):
        return self.vqmodel.quantize.num_embeddings

    @property
    def embed_dim(self):
        return self.vqmodel.quantize.embedding_dim
