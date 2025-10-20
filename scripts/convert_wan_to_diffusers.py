import argparse
import math
import pathlib
from typing import Any, Dict, Tuple

import torch
from accelerate import init_empty_weights
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors.torch import load_file
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    CLIPImageProcessor,
    CLIPVisionConfig,
    CLIPVisionModel,
    CLIPVisionModelWithProjection,
    UMT5EncoderModel,
)

from diffusers import (
    AutoencoderKLWan,
    UniPCMultistepScheduler,
    WanAnimatePipeline,
    WanAnimateTransformer3DModel,
    WanImageToVideoPipeline,
    WanPipeline,
    WanTransformer3DModel,
    WanVACEPipeline,
    WanVACETransformer3DModel,
)


TRANSFORMER_KEYS_RENAME_DICT = {
    "time_embedding.0": "condition_embedder.time_embedder.linear_1",
    "time_embedding.2": "condition_embedder.time_embedder.linear_2",
    "text_embedding.0": "condition_embedder.text_embedder.linear_1",
    "text_embedding.2": "condition_embedder.text_embedder.linear_2",
    "time_projection.1": "condition_embedder.time_proj",
    "head.modulation": "scale_shift_table",
    "head.head": "proj_out",
    "modulation": "scale_shift_table",
    "ffn.0": "ffn.net.0.proj",
    "ffn.2": "ffn.net.2",
    # Hack to swap the layer names
    # The original model calls the norms in following order: norm1, norm3, norm2
    # We convert it to: norm1, norm2, norm3
    "norm2": "norm__placeholder",
    "norm3": "norm2",
    "norm__placeholder": "norm3",
    # For the I2V model
    "img_emb.proj.0": "condition_embedder.image_embedder.norm1",
    "img_emb.proj.1": "condition_embedder.image_embedder.ff.net.0.proj",
    "img_emb.proj.3": "condition_embedder.image_embedder.ff.net.2",
    "img_emb.proj.4": "condition_embedder.image_embedder.norm2",
    # for the FLF2V model
    "img_emb.emb_pos": "condition_embedder.image_embedder.pos_embed",
    # Add attention component mappings
    "self_attn.q": "attn1.to_q",
    "self_attn.k": "attn1.to_k",
    "self_attn.v": "attn1.to_v",
    "self_attn.o": "attn1.to_out.0",
    "self_attn.norm_q": "attn1.norm_q",
    "self_attn.norm_k": "attn1.norm_k",
    "cross_attn.q": "attn2.to_q",
    "cross_attn.k": "attn2.to_k",
    "cross_attn.v": "attn2.to_v",
    "cross_attn.o": "attn2.to_out.0",
    "cross_attn.norm_q": "attn2.norm_q",
    "cross_attn.norm_k": "attn2.norm_k",
    "attn2.to_k_img": "attn2.add_k_proj",
    "attn2.to_v_img": "attn2.add_v_proj",
    "attn2.norm_k_img": "attn2.norm_added_k",
}

VACE_TRANSFORMER_KEYS_RENAME_DICT = {
    "time_embedding.0": "condition_embedder.time_embedder.linear_1",
    "time_embedding.2": "condition_embedder.time_embedder.linear_2",
    "text_embedding.0": "condition_embedder.text_embedder.linear_1",
    "text_embedding.2": "condition_embedder.text_embedder.linear_2",
    "time_projection.1": "condition_embedder.time_proj",
    "head.modulation": "scale_shift_table",
    "head.head": "proj_out",
    "modulation": "scale_shift_table",
    "ffn.0": "ffn.net.0.proj",
    "ffn.2": "ffn.net.2",
    # Hack to swap the layer names
    # The original model calls the norms in following order: norm1, norm3, norm2
    # We convert it to: norm1, norm2, norm3
    "norm2": "norm__placeholder",
    "norm3": "norm2",
    "norm__placeholder": "norm3",
    # # For the I2V model
    # "img_emb.proj.0": "condition_embedder.image_embedder.norm1",
    # "img_emb.proj.1": "condition_embedder.image_embedder.ff.net.0.proj",
    # "img_emb.proj.3": "condition_embedder.image_embedder.ff.net.2",
    # "img_emb.proj.4": "condition_embedder.image_embedder.norm2",
    # # for the FLF2V model
    # "img_emb.emb_pos": "condition_embedder.image_embedder.pos_embed",
    # Add attention component mappings
    "self_attn.q": "attn1.to_q",
    "self_attn.k": "attn1.to_k",
    "self_attn.v": "attn1.to_v",
    "self_attn.o": "attn1.to_out.0",
    "self_attn.norm_q": "attn1.norm_q",
    "self_attn.norm_k": "attn1.norm_k",
    "cross_attn.q": "attn2.to_q",
    "cross_attn.k": "attn2.to_k",
    "cross_attn.v": "attn2.to_v",
    "cross_attn.o": "attn2.to_out.0",
    "cross_attn.norm_q": "attn2.norm_q",
    "cross_attn.norm_k": "attn2.norm_k",
    "attn2.to_k_img": "attn2.add_k_proj",
    "attn2.to_v_img": "attn2.add_v_proj",
    "attn2.norm_k_img": "attn2.norm_added_k",
    "before_proj": "proj_in",
    "after_proj": "proj_out",
}

ANIMATE_TRANSFORMER_KEYS_RENAME_DICT = {
    "time_embedding.0": "condition_embedder.time_embedder.linear_1",
    "time_embedding.2": "condition_embedder.time_embedder.linear_2",
    "text_embedding.0": "condition_embedder.text_embedder.linear_1",
    "text_embedding.2": "condition_embedder.text_embedder.linear_2",
    "time_projection.1": "condition_embedder.time_proj",
    "head.modulation": "scale_shift_table",
    "head.head": "proj_out",
    "modulation": "scale_shift_table",
    "ffn.0": "ffn.net.0.proj",
    "ffn.2": "ffn.net.2",
    # Hack to swap the layer names
    # The original model calls the norms in following order: norm1, norm3, norm2
    # We convert it to: norm1, norm2, norm3
    "norm2": "norm__placeholder",
    "norm3": "norm2",
    "norm__placeholder": "norm3",
    "img_emb.proj.0": "condition_embedder.image_embedder.norm1",
    "img_emb.proj.1": "condition_embedder.image_embedder.ff.net.0.proj",
    "img_emb.proj.3": "condition_embedder.image_embedder.ff.net.2",
    "img_emb.proj.4": "condition_embedder.image_embedder.norm2",
    # Add attention component mappings
    "self_attn.q": "attn1.to_q",
    "self_attn.k": "attn1.to_k",
    "self_attn.v": "attn1.to_v",
    "self_attn.o": "attn1.to_out.0",
    "self_attn.norm_q": "attn1.norm_q",
    "self_attn.norm_k": "attn1.norm_k",
    "cross_attn.q": "attn2.to_q",
    "cross_attn.k": "attn2.to_k",
    "cross_attn.v": "attn2.to_v",
    "cross_attn.o": "attn2.to_out.0",
    "cross_attn.norm_q": "attn2.norm_q",
    "cross_attn.norm_k": "attn2.norm_k",
    "cross_attn.k_img": "attn2.to_k_img",
    "cross_attn.v_img": "attn2.to_v_img",
    "cross_attn.norm_k_img": "attn2.norm_k_img",
    # After cross_attn -> attn2 rename, we need to rename the img keys
    "attn2.to_k_img": "attn2.add_k_proj",
    "attn2.to_v_img": "attn2.add_v_proj",
    "attn2.norm_k_img": "attn2.norm_added_k",
    # Motion encoder mappings
    "motion_encoder.enc.net_app.convs": "condition_embedder.motion_embedder.convs",
    "motion_encoder.enc.fc": "condition_embedder.motion_embedder.linears",
    "motion_encoder.dec.direction.weight": "condition_embedder.motion_embedder.motion_synthesis_weight",
    # Face encoder mappings - CausalConv1d has a .conv submodule that we need to flatten
    "face_encoder.conv1_local.conv": "condition_embedder.face_embedder.conv1_local",
    "face_encoder.conv2.conv": "condition_embedder.face_embedder.conv2",
    "face_encoder.conv3.conv": "condition_embedder.face_embedder.conv3",
    "face_encoder.out_proj": "condition_embedder.face_embedder.out_proj",
    "face_encoder.norm1": "condition_embedder.face_embedder.norm1",
    "face_encoder.norm2": "condition_embedder.face_embedder.norm2",
    "face_encoder.norm3": "condition_embedder.face_embedder.norm3",
    "face_encoder.padding_tokens": "condition_embedder.face_embedder.padding_tokens",
    # Face adapter mappings
    "face_adapter.fuser_blocks": "face_adapter",
}


def convert_equal_linear_weight(key: str, state_dict: Dict[str, Any]) -> None:
    """
    Convert EqualLinear weights to standard Linear weights by applying the scale factor.
    EqualLinear uses: F.linear(input, self.weight * self.scale, bias=self.bias)
    where scale = (1 / sqrt(in_dim))
    """
    if ".weight" not in key:
        return

    in_dim = state_dict[key].shape[1]
    scale = 1.0 / math.sqrt(in_dim)
    state_dict[key] = state_dict[key] * scale


def convert_equal_conv2d_weight(key: str, state_dict: Dict[str, Any]) -> None:
    """
    Convert EqualConv2d weights to standard Conv2d weights by applying the scale factor.
    EqualConv2d uses: F.conv2d(input, self.weight * self.scale, bias=self.bias, ...)
    where scale = 1 / sqrt(in_channel * kernel_size^2)
    """
    if ".weight" not in key or len(state_dict[key].shape) != 4:
        return

    out_channel, in_channel, kernel_size, kernel_size = state_dict[key].shape
    scale = 1.0 / math.sqrt(in_channel * kernel_size**2)
    state_dict[key] = state_dict[key] * scale


def convert_animate_motion_encoder_weights(key: str, state_dict: Dict[str, Any]) -> None:
    """
    Convert all motion encoder weights for Animate model.
    This handles both EqualLinear (in linears) and EqualConv2d (in convs).

    In the original model:
    - All Linear layers in fc use EqualLinear
    - All Conv2d layers in convs use EqualConv2d (except blur_conv which is initialized separately)
    - Blur kernels are stored as buffers in Sequential modules
    - ConvLayer is nn.Sequential with indices: [Blur (optional), EqualConv2d, FusedLeakyReLU (optional)]

    Conversion strategy:
    1. Drop .kernel buffers (blur kernels)
    2. Rename sequential indices to named components (e.g., 0 -> conv2d, 1 -> bias_leaky_relu)
    3. Scale EqualLinear and EqualConv2d weights
    """


TRANSFORMER_SPECIAL_KEYS_REMAP = {}
VACE_TRANSFORMER_SPECIAL_KEYS_REMAP = {}
ANIMATE_TRANSFORMER_SPECIAL_KEYS_REMAP = {"condition_embedder.motion_embedder": convert_animate_motion_encoder_weights}


def update_state_dict_(state_dict: Dict[str, Any], old_key: str, new_key: str) -> Dict[str, Any]:
    state_dict[new_key] = state_dict.pop(old_key)


def load_sharded_safetensors(dir: pathlib.Path):
    file_paths = list(dir.glob("diffusion_pytorch_model*.safetensors"))
    state_dict = {}
    for path in file_paths:
        state_dict.update(load_file(path))
    return state_dict


def get_transformer_config(model_type: str) -> Tuple[Dict[str, Any], ...]:
    if model_type == "Wan-T2V-1.3B":
        config = {
            "model_id": "StevenZhang/Wan2.1-T2V-1.3B-Diff",
            "diffusers_config": {
                "added_kv_proj_dim": None,
                "attention_head_dim": 128,
                "cross_attn_norm": True,
                "eps": 1e-06,
                "ffn_dim": 8960,
                "freq_dim": 256,
                "in_channels": 16,
                "num_attention_heads": 12,
                "num_layers": 30,
                "out_channels": 16,
                "patch_size": [1, 2, 2],
                "qk_norm": "rms_norm_across_heads",
                "text_dim": 4096,
            },
        }
        RENAME_DICT = TRANSFORMER_KEYS_RENAME_DICT
        SPECIAL_KEYS_REMAP = TRANSFORMER_SPECIAL_KEYS_REMAP
    elif model_type == "Wan-T2V-14B":
        config = {
            "model_id": "StevenZhang/Wan2.1-T2V-14B-Diff",
            "diffusers_config": {
                "added_kv_proj_dim": None,
                "attention_head_dim": 128,
                "cross_attn_norm": True,
                "eps": 1e-06,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "in_channels": 16,
                "num_attention_heads": 40,
                "num_layers": 40,
                "out_channels": 16,
                "patch_size": [1, 2, 2],
                "qk_norm": "rms_norm_across_heads",
                "text_dim": 4096,
            },
        }
        RENAME_DICT = TRANSFORMER_KEYS_RENAME_DICT
        SPECIAL_KEYS_REMAP = TRANSFORMER_SPECIAL_KEYS_REMAP
    elif model_type == "Wan-I2V-14B-480p":
        config = {
            "model_id": "StevenZhang/Wan2.1-I2V-14B-480P-Diff",
            "diffusers_config": {
                "image_dim": 1280,
                "added_kv_proj_dim": 5120,
                "attention_head_dim": 128,
                "cross_attn_norm": True,
                "eps": 1e-06,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "in_channels": 36,
                "num_attention_heads": 40,
                "num_layers": 40,
                "out_channels": 16,
                "patch_size": [1, 2, 2],
                "qk_norm": "rms_norm_across_heads",
                "text_dim": 4096,
            },
        }
        RENAME_DICT = TRANSFORMER_KEYS_RENAME_DICT
        SPECIAL_KEYS_REMAP = TRANSFORMER_SPECIAL_KEYS_REMAP
    elif model_type == "Wan-I2V-14B-720p":
        config = {
            "model_id": "StevenZhang/Wan2.1-I2V-14B-720P-Diff",
            "diffusers_config": {
                "image_dim": 1280,
                "added_kv_proj_dim": 5120,
                "attention_head_dim": 128,
                "cross_attn_norm": True,
                "eps": 1e-06,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "in_channels": 36,
                "num_attention_heads": 40,
                "num_layers": 40,
                "out_channels": 16,
                "patch_size": [1, 2, 2],
                "qk_norm": "rms_norm_across_heads",
                "text_dim": 4096,
            },
        }
        RENAME_DICT = TRANSFORMER_KEYS_RENAME_DICT
        SPECIAL_KEYS_REMAP = TRANSFORMER_SPECIAL_KEYS_REMAP
    elif model_type == "Wan-FLF2V-14B-720P":
        config = {
            "model_id": "ypyp/Wan2.1-FLF2V-14B-720P",  # This is just a placeholder
            "diffusers_config": {
                "image_dim": 1280,
                "added_kv_proj_dim": 5120,
                "attention_head_dim": 128,
                "cross_attn_norm": True,
                "eps": 1e-06,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "in_channels": 36,
                "num_attention_heads": 40,
                "num_layers": 40,
                "out_channels": 16,
                "patch_size": [1, 2, 2],
                "qk_norm": "rms_norm_across_heads",
                "text_dim": 4096,
                "rope_max_seq_len": 1024,
                "pos_embed_seq_len": 257 * 2,
            },
        }
        RENAME_DICT = TRANSFORMER_KEYS_RENAME_DICT
        SPECIAL_KEYS_REMAP = TRANSFORMER_SPECIAL_KEYS_REMAP
    elif model_type == "Wan-VACE-1.3B":
        config = {
            "model_id": "Wan-AI/Wan2.1-VACE-1.3B",
            "diffusers_config": {
                "added_kv_proj_dim": None,
                "attention_head_dim": 128,
                "cross_attn_norm": True,
                "eps": 1e-06,
                "ffn_dim": 8960,
                "freq_dim": 256,
                "in_channels": 16,
                "num_attention_heads": 12,
                "num_layers": 30,
                "out_channels": 16,
                "patch_size": [1, 2, 2],
                "qk_norm": "rms_norm_across_heads",
                "text_dim": 4096,
                "vace_layers": [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28],
                "vace_in_channels": 96,
            },
        }
        RENAME_DICT = VACE_TRANSFORMER_KEYS_RENAME_DICT
        SPECIAL_KEYS_REMAP = VACE_TRANSFORMER_SPECIAL_KEYS_REMAP
    elif model_type == "Wan-VACE-14B":
        config = {
            "model_id": "Wan-AI/Wan2.1-VACE-14B",
            "diffusers_config": {
                "added_kv_proj_dim": None,
                "attention_head_dim": 128,
                "cross_attn_norm": True,
                "eps": 1e-06,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "in_channels": 16,
                "num_attention_heads": 40,
                "num_layers": 40,
                "out_channels": 16,
                "patch_size": [1, 2, 2],
                "qk_norm": "rms_norm_across_heads",
                "text_dim": 4096,
                "vace_layers": [0, 5, 10, 15, 20, 25, 30, 35],
                "vace_in_channels": 96,
            },
        }
        RENAME_DICT = VACE_TRANSFORMER_KEYS_RENAME_DICT
        SPECIAL_KEYS_REMAP = VACE_TRANSFORMER_SPECIAL_KEYS_REMAP
    elif model_type == "Wan2.2-VACE-Fun-14B":
        config = {
            "model_id": "alibaba-pai/Wan2.2-VACE-Fun-A14B",
            "diffusers_config": {
                "added_kv_proj_dim": None,
                "attention_head_dim": 128,
                "cross_attn_norm": True,
                "eps": 1e-06,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "in_channels": 16,
                "num_attention_heads": 40,
                "num_layers": 40,
                "out_channels": 16,
                "patch_size": [1, 2, 2],
                "qk_norm": "rms_norm_across_heads",
                "text_dim": 4096,
                "vace_layers": [0, 5, 10, 15, 20, 25, 30, 35],
                "vace_in_channels": 96,
            },
        }
        RENAME_DICT = VACE_TRANSFORMER_KEYS_RENAME_DICT
        SPECIAL_KEYS_REMAP = VACE_TRANSFORMER_SPECIAL_KEYS_REMAP
    elif model_type == "Wan2.2-I2V-14B-720p":
        config = {
            "model_id": "Wan-AI/Wan2.2-I2V-A14B",
            "diffusers_config": {
                "added_kv_proj_dim": None,
                "attention_head_dim": 128,
                "cross_attn_norm": True,
                "eps": 1e-06,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "in_channels": 36,
                "num_attention_heads": 40,
                "num_layers": 40,
                "out_channels": 16,
                "patch_size": [1, 2, 2],
                "qk_norm": "rms_norm_across_heads",
                "text_dim": 4096,
            },
        }
        RENAME_DICT = TRANSFORMER_KEYS_RENAME_DICT
        SPECIAL_KEYS_REMAP = TRANSFORMER_SPECIAL_KEYS_REMAP
    elif model_type == "Wan2.2-T2V-A14B":
        config = {
            "model_id": "Wan-AI/Wan2.2-T2V-A14B",
            "diffusers_config": {
                "added_kv_proj_dim": None,
                "attention_head_dim": 128,
                "cross_attn_norm": True,
                "eps": 1e-06,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "in_channels": 16,
                "num_attention_heads": 40,
                "num_layers": 40,
                "out_channels": 16,
                "patch_size": [1, 2, 2],
                "qk_norm": "rms_norm_across_heads",
                "text_dim": 4096,
            },
        }
        RENAME_DICT = TRANSFORMER_KEYS_RENAME_DICT
        SPECIAL_KEYS_REMAP = TRANSFORMER_SPECIAL_KEYS_REMAP
    elif model_type == "Wan2.2-TI2V-5B":
        config = {
            "model_id": "Wan-AI/Wan2.2-TI2V-5B",
            "diffusers_config": {
                "added_kv_proj_dim": None,
                "attention_head_dim": 128,
                "cross_attn_norm": True,
                "eps": 1e-06,
                "ffn_dim": 14336,
                "freq_dim": 256,
                "in_channels": 48,
                "num_attention_heads": 24,
                "num_layers": 30,
                "out_channels": 48,
                "patch_size": [1, 2, 2],
                "qk_norm": "rms_norm_across_heads",
                "text_dim": 4096,
            },
        }
        RENAME_DICT = TRANSFORMER_KEYS_RENAME_DICT
        SPECIAL_KEYS_REMAP = TRANSFORMER_SPECIAL_KEYS_REMAP
    elif model_type == "Wan2.2-Animate-14B":
        config = {
            "model_id": "Wan-AI/Wan2.2-Animate-14B",
            "diffusers_config": {
                "image_dim": 1280,
                "added_kv_proj_dim": 5120,
                "attention_head_dim": 128,
                "cross_attn_norm": True,
                "eps": 1e-06,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "in_channels": 36,
                "motion_encoder_dim": 512,
                "num_attention_heads": 40,
                "num_layers": 40,
                "out_channels": 16,
                "patch_size": (1, 2, 2),
                "qk_norm": "rms_norm_across_heads",
                "text_dim": 4096,
                "rope_max_seq_len": 1024,
                "pos_embed_seq_len": 257 * 2,
            },
        }
        RENAME_DICT = ANIMATE_TRANSFORMER_KEYS_RENAME_DICT
        SPECIAL_KEYS_REMAP = ANIMATE_TRANSFORMER_SPECIAL_KEYS_REMAP
    return config, RENAME_DICT, SPECIAL_KEYS_REMAP


def convert_transformer(model_type: str, stage: str = None):
    config, RENAME_DICT, SPECIAL_KEYS_REMAP = get_transformer_config(model_type)

    diffusers_config = config["diffusers_config"]
    model_id = config["model_id"]
    model_dir = pathlib.Path(snapshot_download(model_id, repo_type="model"))

    if stage is not None:
        model_dir = model_dir / stage

    original_state_dict = load_sharded_safetensors(model_dir)

    with init_empty_weights():
        if "Animate" in model_type:
            transformer = WanAnimateTransformer3DModel.from_config(diffusers_config)
        elif "VACE" in model_type:
            transformer = WanVACETransformer3DModel.from_config(diffusers_config)
        else:
            transformer = WanTransformer3DModel.from_config(diffusers_config)

    for key in list(original_state_dict.keys()):
        new_key = key[:]
        for replace_key, rename_key in RENAME_DICT.items():
            new_key = new_key.replace(replace_key, rename_key)
        update_state_dict_(original_state_dict, key, new_key)

    for key in list(original_state_dict.keys()):
        for special_key, handler_fn_inplace in SPECIAL_KEYS_REMAP.items():
            if special_key not in key:
                continue
            handler_fn_inplace(key, original_state_dict)

    # For Animate model, add blur_conv weights from the initialized model
    # These are procedurally generated in the diffusers ConvLayer and not present in original checkpoint
    if "Animate" in model_type:
        # Create a temporary model on CPU to get the blur_conv weights
        with torch.device("cpu"):
            temp_transformer = WanAnimateTransformer3DModel.from_config(diffusers_config)
        temp_model_state = temp_transformer.state_dict()
        for key in temp_model_state.keys():
            if "blur_conv.weight" in key and "motion_embedder" in key:
                original_state_dict[key] = temp_model_state[key]
        del temp_transformer

    # Load state dict into the meta model, which will materialize the tensors
    transformer.load_state_dict(original_state_dict, strict=True, assign=True)

    # Move to CPU to ensure all tensors are materialized
    transformer = transformer.to("cpu")

    return transformer


def convert_vae():
    vae_ckpt_path = hf_hub_download("Wan-AI/Wan2.1-T2V-14B", "Wan2.1_VAE.pth")
    old_state_dict = torch.load(vae_ckpt_path, weights_only=True)
    new_state_dict = {}

    # Create mappings for specific components
    middle_key_mapping = {
        # Encoder middle block
        "encoder.middle.0.residual.0.gamma": "encoder.mid_block.resnets.0.norm1.gamma",
        "encoder.middle.0.residual.2.bias": "encoder.mid_block.resnets.0.conv1.bias",
        "encoder.middle.0.residual.2.weight": "encoder.mid_block.resnets.0.conv1.weight",
        "encoder.middle.0.residual.3.gamma": "encoder.mid_block.resnets.0.norm2.gamma",
        "encoder.middle.0.residual.6.bias": "encoder.mid_block.resnets.0.conv2.bias",
        "encoder.middle.0.residual.6.weight": "encoder.mid_block.resnets.0.conv2.weight",
        "encoder.middle.2.residual.0.gamma": "encoder.mid_block.resnets.1.norm1.gamma",
        "encoder.middle.2.residual.2.bias": "encoder.mid_block.resnets.1.conv1.bias",
        "encoder.middle.2.residual.2.weight": "encoder.mid_block.resnets.1.conv1.weight",
        "encoder.middle.2.residual.3.gamma": "encoder.mid_block.resnets.1.norm2.gamma",
        "encoder.middle.2.residual.6.bias": "encoder.mid_block.resnets.1.conv2.bias",
        "encoder.middle.2.residual.6.weight": "encoder.mid_block.resnets.1.conv2.weight",
        # Decoder middle block
        "decoder.middle.0.residual.0.gamma": "decoder.mid_block.resnets.0.norm1.gamma",
        "decoder.middle.0.residual.2.bias": "decoder.mid_block.resnets.0.conv1.bias",
        "decoder.middle.0.residual.2.weight": "decoder.mid_block.resnets.0.conv1.weight",
        "decoder.middle.0.residual.3.gamma": "decoder.mid_block.resnets.0.norm2.gamma",
        "decoder.middle.0.residual.6.bias": "decoder.mid_block.resnets.0.conv2.bias",
        "decoder.middle.0.residual.6.weight": "decoder.mid_block.resnets.0.conv2.weight",
        "decoder.middle.2.residual.0.gamma": "decoder.mid_block.resnets.1.norm1.gamma",
        "decoder.middle.2.residual.2.bias": "decoder.mid_block.resnets.1.conv1.bias",
        "decoder.middle.2.residual.2.weight": "decoder.mid_block.resnets.1.conv1.weight",
        "decoder.middle.2.residual.3.gamma": "decoder.mid_block.resnets.1.norm2.gamma",
        "decoder.middle.2.residual.6.bias": "decoder.mid_block.resnets.1.conv2.bias",
        "decoder.middle.2.residual.6.weight": "decoder.mid_block.resnets.1.conv2.weight",
    }

    # Create a mapping for attention blocks
    attention_mapping = {
        # Encoder middle attention
        "encoder.middle.1.norm.gamma": "encoder.mid_block.attentions.0.norm.gamma",
        "encoder.middle.1.to_qkv.weight": "encoder.mid_block.attentions.0.to_qkv.weight",
        "encoder.middle.1.to_qkv.bias": "encoder.mid_block.attentions.0.to_qkv.bias",
        "encoder.middle.1.proj.weight": "encoder.mid_block.attentions.0.proj.weight",
        "encoder.middle.1.proj.bias": "encoder.mid_block.attentions.0.proj.bias",
        # Decoder middle attention
        "decoder.middle.1.norm.gamma": "decoder.mid_block.attentions.0.norm.gamma",
        "decoder.middle.1.to_qkv.weight": "decoder.mid_block.attentions.0.to_qkv.weight",
        "decoder.middle.1.to_qkv.bias": "decoder.mid_block.attentions.0.to_qkv.bias",
        "decoder.middle.1.proj.weight": "decoder.mid_block.attentions.0.proj.weight",
        "decoder.middle.1.proj.bias": "decoder.mid_block.attentions.0.proj.bias",
    }

    # Create a mapping for the head components
    head_mapping = {
        # Encoder head
        "encoder.head.0.gamma": "encoder.norm_out.gamma",
        "encoder.head.2.bias": "encoder.conv_out.bias",
        "encoder.head.2.weight": "encoder.conv_out.weight",
        # Decoder head
        "decoder.head.0.gamma": "decoder.norm_out.gamma",
        "decoder.head.2.bias": "decoder.conv_out.bias",
        "decoder.head.2.weight": "decoder.conv_out.weight",
    }

    # Create a mapping for the quant components
    quant_mapping = {
        "conv1.weight": "quant_conv.weight",
        "conv1.bias": "quant_conv.bias",
        "conv2.weight": "post_quant_conv.weight",
        "conv2.bias": "post_quant_conv.bias",
    }

    # Process each key in the state dict
    for key, value in old_state_dict.items():
        # Handle middle block keys using the mapping
        if key in middle_key_mapping:
            new_key = middle_key_mapping[key]
            new_state_dict[new_key] = value
        # Handle attention blocks using the mapping
        elif key in attention_mapping:
            new_key = attention_mapping[key]
            new_state_dict[new_key] = value
        # Handle head keys using the mapping
        elif key in head_mapping:
            new_key = head_mapping[key]
            new_state_dict[new_key] = value
        # Handle quant keys using the mapping
        elif key in quant_mapping:
            new_key = quant_mapping[key]
            new_state_dict[new_key] = value
        # Handle encoder conv1
        elif key == "encoder.conv1.weight":
            new_state_dict["encoder.conv_in.weight"] = value
        elif key == "encoder.conv1.bias":
            new_state_dict["encoder.conv_in.bias"] = value
        # Handle decoder conv1
        elif key == "decoder.conv1.weight":
            new_state_dict["decoder.conv_in.weight"] = value
        elif key == "decoder.conv1.bias":
            new_state_dict["decoder.conv_in.bias"] = value
        # Handle encoder downsamples
        elif key.startswith("encoder.downsamples."):
            # Convert to down_blocks
            new_key = key.replace("encoder.downsamples.", "encoder.down_blocks.")

            # Convert residual block naming but keep the original structure
            if ".residual.0.gamma" in new_key:
                new_key = new_key.replace(".residual.0.gamma", ".norm1.gamma")
            elif ".residual.2.bias" in new_key:
                new_key = new_key.replace(".residual.2.bias", ".conv1.bias")
            elif ".residual.2.weight" in new_key:
                new_key = new_key.replace(".residual.2.weight", ".conv1.weight")
            elif ".residual.3.gamma" in new_key:
                new_key = new_key.replace(".residual.3.gamma", ".norm2.gamma")
            elif ".residual.6.bias" in new_key:
                new_key = new_key.replace(".residual.6.bias", ".conv2.bias")
            elif ".residual.6.weight" in new_key:
                new_key = new_key.replace(".residual.6.weight", ".conv2.weight")
            elif ".shortcut.bias" in new_key:
                new_key = new_key.replace(".shortcut.bias", ".conv_shortcut.bias")
            elif ".shortcut.weight" in new_key:
                new_key = new_key.replace(".shortcut.weight", ".conv_shortcut.weight")

            new_state_dict[new_key] = value

        # Handle decoder upsamples
        elif key.startswith("decoder.upsamples."):
            # Convert to up_blocks
            parts = key.split(".")
            block_idx = int(parts[2])

            # Group residual blocks
            if "residual" in key:
                if block_idx in [0, 1, 2]:
                    new_block_idx = 0
                    resnet_idx = block_idx
                elif block_idx in [4, 5, 6]:
                    new_block_idx = 1
                    resnet_idx = block_idx - 4
                elif block_idx in [8, 9, 10]:
                    new_block_idx = 2
                    resnet_idx = block_idx - 8
                elif block_idx in [12, 13, 14]:
                    new_block_idx = 3
                    resnet_idx = block_idx - 12
                else:
                    # Keep as is for other blocks
                    new_state_dict[key] = value
                    continue

                # Convert residual block naming
                if ".residual.0.gamma" in key:
                    new_key = f"decoder.up_blocks.{new_block_idx}.resnets.{resnet_idx}.norm1.gamma"
                elif ".residual.2.bias" in key:
                    new_key = f"decoder.up_blocks.{new_block_idx}.resnets.{resnet_idx}.conv1.bias"
                elif ".residual.2.weight" in key:
                    new_key = f"decoder.up_blocks.{new_block_idx}.resnets.{resnet_idx}.conv1.weight"
                elif ".residual.3.gamma" in key:
                    new_key = f"decoder.up_blocks.{new_block_idx}.resnets.{resnet_idx}.norm2.gamma"
                elif ".residual.6.bias" in key:
                    new_key = f"decoder.up_blocks.{new_block_idx}.resnets.{resnet_idx}.conv2.bias"
                elif ".residual.6.weight" in key:
                    new_key = f"decoder.up_blocks.{new_block_idx}.resnets.{resnet_idx}.conv2.weight"
                else:
                    new_key = key

                new_state_dict[new_key] = value

            # Handle shortcut connections
            elif ".shortcut." in key:
                if block_idx == 4:
                    new_key = key.replace(".shortcut.", ".resnets.0.conv_shortcut.")
                    new_key = new_key.replace("decoder.upsamples.4", "decoder.up_blocks.1")
                else:
                    new_key = key.replace("decoder.upsamples.", "decoder.up_blocks.")
                    new_key = new_key.replace(".shortcut.", ".conv_shortcut.")

                new_state_dict[new_key] = value

            # Handle upsamplers
            elif ".resample." in key or ".time_conv." in key:
                if block_idx == 3:
                    new_key = key.replace(f"decoder.upsamples.{block_idx}", "decoder.up_blocks.0.upsamplers.0")
                elif block_idx == 7:
                    new_key = key.replace(f"decoder.upsamples.{block_idx}", "decoder.up_blocks.1.upsamplers.0")
                elif block_idx == 11:
                    new_key = key.replace(f"decoder.upsamples.{block_idx}", "decoder.up_blocks.2.upsamplers.0")
                else:
                    new_key = key.replace("decoder.upsamples.", "decoder.up_blocks.")

                new_state_dict[new_key] = value
            else:
                new_key = key.replace("decoder.upsamples.", "decoder.up_blocks.")
                new_state_dict[new_key] = value
        else:
            # Keep other keys unchanged
            new_state_dict[key] = value

    with init_empty_weights():
        vae = AutoencoderKLWan()
    vae.load_state_dict(new_state_dict, strict=True, assign=True)
    return vae


vae22_diffusers_config = {
    "base_dim": 160,
    "z_dim": 48,
    "is_residual": True,
    "in_channels": 12,
    "out_channels": 12,
    "decoder_base_dim": 256,
    "scale_factor_temporal": 4,
    "scale_factor_spatial": 16,
    "patch_size": 2,
    "latents_mean": [
        -0.2289,
        -0.0052,
        -0.1323,
        -0.2339,
        -0.2799,
        0.0174,
        0.1838,
        0.1557,
        -0.1382,
        0.0542,
        0.2813,
        0.0891,
        0.1570,
        -0.0098,
        0.0375,
        -0.1825,
        -0.2246,
        -0.1207,
        -0.0698,
        0.5109,
        0.2665,
        -0.2108,
        -0.2158,
        0.2502,
        -0.2055,
        -0.0322,
        0.1109,
        0.1567,
        -0.0729,
        0.0899,
        -0.2799,
        -0.1230,
        -0.0313,
        -0.1649,
        0.0117,
        0.0723,
        -0.2839,
        -0.2083,
        -0.0520,
        0.3748,
        0.0152,
        0.1957,
        0.1433,
        -0.2944,
        0.3573,
        -0.0548,
        -0.1681,
        -0.0667,
    ],
    "latents_std": [
        0.4765,
        1.0364,
        0.4514,
        1.1677,
        0.5313,
        0.4990,
        0.4818,
        0.5013,
        0.8158,
        1.0344,
        0.5894,
        1.0901,
        0.6885,
        0.6165,
        0.8454,
        0.4978,
        0.5759,
        0.3523,
        0.7135,
        0.6804,
        0.5833,
        1.4146,
        0.8986,
        0.5659,
        0.7069,
        0.5338,
        0.4889,
        0.4917,
        0.4069,
        0.4999,
        0.6866,
        0.4093,
        0.5709,
        0.6065,
        0.6415,
        0.4944,
        0.5726,
        1.2042,
        0.5458,
        1.6887,
        0.3971,
        1.0600,
        0.3943,
        0.5537,
        0.5444,
        0.4089,
        0.7468,
        0.7744,
    ],
    "clip_output": False,
}


def convert_vae_22():
    vae_ckpt_path = hf_hub_download("Wan-AI/Wan2.2-TI2V-5B", "Wan2.2_VAE.pth")
    old_state_dict = torch.load(vae_ckpt_path, weights_only=True)
    new_state_dict = {}

    # Create mappings for specific components
    middle_key_mapping = {
        # Encoder middle block
        "encoder.middle.0.residual.0.gamma": "encoder.mid_block.resnets.0.norm1.gamma",
        "encoder.middle.0.residual.2.bias": "encoder.mid_block.resnets.0.conv1.bias",
        "encoder.middle.0.residual.2.weight": "encoder.mid_block.resnets.0.conv1.weight",
        "encoder.middle.0.residual.3.gamma": "encoder.mid_block.resnets.0.norm2.gamma",
        "encoder.middle.0.residual.6.bias": "encoder.mid_block.resnets.0.conv2.bias",
        "encoder.middle.0.residual.6.weight": "encoder.mid_block.resnets.0.conv2.weight",
        "encoder.middle.2.residual.0.gamma": "encoder.mid_block.resnets.1.norm1.gamma",
        "encoder.middle.2.residual.2.bias": "encoder.mid_block.resnets.1.conv1.bias",
        "encoder.middle.2.residual.2.weight": "encoder.mid_block.resnets.1.conv1.weight",
        "encoder.middle.2.residual.3.gamma": "encoder.mid_block.resnets.1.norm2.gamma",
        "encoder.middle.2.residual.6.bias": "encoder.mid_block.resnets.1.conv2.bias",
        "encoder.middle.2.residual.6.weight": "encoder.mid_block.resnets.1.conv2.weight",
        # Decoder middle block
        "decoder.middle.0.residual.0.gamma": "decoder.mid_block.resnets.0.norm1.gamma",
        "decoder.middle.0.residual.2.bias": "decoder.mid_block.resnets.0.conv1.bias",
        "decoder.middle.0.residual.2.weight": "decoder.mid_block.resnets.0.conv1.weight",
        "decoder.middle.0.residual.3.gamma": "decoder.mid_block.resnets.0.norm2.gamma",
        "decoder.middle.0.residual.6.bias": "decoder.mid_block.resnets.0.conv2.bias",
        "decoder.middle.0.residual.6.weight": "decoder.mid_block.resnets.0.conv2.weight",
        "decoder.middle.2.residual.0.gamma": "decoder.mid_block.resnets.1.norm1.gamma",
        "decoder.middle.2.residual.2.bias": "decoder.mid_block.resnets.1.conv1.bias",
        "decoder.middle.2.residual.2.weight": "decoder.mid_block.resnets.1.conv1.weight",
        "decoder.middle.2.residual.3.gamma": "decoder.mid_block.resnets.1.norm2.gamma",
        "decoder.middle.2.residual.6.bias": "decoder.mid_block.resnets.1.conv2.bias",
        "decoder.middle.2.residual.6.weight": "decoder.mid_block.resnets.1.conv2.weight",
    }

    # Create a mapping for attention blocks
    attention_mapping = {
        # Encoder middle attention
        "encoder.middle.1.norm.gamma": "encoder.mid_block.attentions.0.norm.gamma",
        "encoder.middle.1.to_qkv.weight": "encoder.mid_block.attentions.0.to_qkv.weight",
        "encoder.middle.1.to_qkv.bias": "encoder.mid_block.attentions.0.to_qkv.bias",
        "encoder.middle.1.proj.weight": "encoder.mid_block.attentions.0.proj.weight",
        "encoder.middle.1.proj.bias": "encoder.mid_block.attentions.0.proj.bias",
        # Decoder middle attention
        "decoder.middle.1.norm.gamma": "decoder.mid_block.attentions.0.norm.gamma",
        "decoder.middle.1.to_qkv.weight": "decoder.mid_block.attentions.0.to_qkv.weight",
        "decoder.middle.1.to_qkv.bias": "decoder.mid_block.attentions.0.to_qkv.bias",
        "decoder.middle.1.proj.weight": "decoder.mid_block.attentions.0.proj.weight",
        "decoder.middle.1.proj.bias": "decoder.mid_block.attentions.0.proj.bias",
    }

    # Create a mapping for the head components
    head_mapping = {
        # Encoder head
        "encoder.head.0.gamma": "encoder.norm_out.gamma",
        "encoder.head.2.bias": "encoder.conv_out.bias",
        "encoder.head.2.weight": "encoder.conv_out.weight",
        # Decoder head
        "decoder.head.0.gamma": "decoder.norm_out.gamma",
        "decoder.head.2.bias": "decoder.conv_out.bias",
        "decoder.head.2.weight": "decoder.conv_out.weight",
    }

    # Create a mapping for the quant components
    quant_mapping = {
        "conv1.weight": "quant_conv.weight",
        "conv1.bias": "quant_conv.bias",
        "conv2.weight": "post_quant_conv.weight",
        "conv2.bias": "post_quant_conv.bias",
    }

    # Process each key in the state dict
    for key, value in old_state_dict.items():
        # Handle middle block keys using the mapping
        if key in middle_key_mapping:
            new_key = middle_key_mapping[key]
            new_state_dict[new_key] = value
        # Handle attention blocks using the mapping
        elif key in attention_mapping:
            new_key = attention_mapping[key]
            new_state_dict[new_key] = value
        # Handle head keys using the mapping
        elif key in head_mapping:
            new_key = head_mapping[key]
            new_state_dict[new_key] = value
        # Handle quant keys using the mapping
        elif key in quant_mapping:
            new_key = quant_mapping[key]
            new_state_dict[new_key] = value
        # Handle encoder conv1
        elif key == "encoder.conv1.weight":
            new_state_dict["encoder.conv_in.weight"] = value
        elif key == "encoder.conv1.bias":
            new_state_dict["encoder.conv_in.bias"] = value
        # Handle decoder conv1
        elif key == "decoder.conv1.weight":
            new_state_dict["decoder.conv_in.weight"] = value
        elif key == "decoder.conv1.bias":
            new_state_dict["decoder.conv_in.bias"] = value
        # Handle encoder downsamples
        elif key.startswith("encoder.downsamples."):
            # Change encoder.downsamples to encoder.down_blocks
            new_key = key.replace("encoder.downsamples.", "encoder.down_blocks.")

            # Handle residual blocks - change downsamples to resnets and rename components
            if "residual" in new_key or "shortcut" in new_key:
                # Change the second downsamples to resnets
                new_key = new_key.replace(".downsamples.", ".resnets.")

                # Rename residual components
                if ".residual.0.gamma" in new_key:
                    new_key = new_key.replace(".residual.0.gamma", ".norm1.gamma")
                elif ".residual.2.weight" in new_key:
                    new_key = new_key.replace(".residual.2.weight", ".conv1.weight")
                elif ".residual.2.bias" in new_key:
                    new_key = new_key.replace(".residual.2.bias", ".conv1.bias")
                elif ".residual.3.gamma" in new_key:
                    new_key = new_key.replace(".residual.3.gamma", ".norm2.gamma")
                elif ".residual.6.weight" in new_key:
                    new_key = new_key.replace(".residual.6.weight", ".conv2.weight")
                elif ".residual.6.bias" in new_key:
                    new_key = new_key.replace(".residual.6.bias", ".conv2.bias")
                elif ".shortcut.weight" in new_key:
                    new_key = new_key.replace(".shortcut.weight", ".conv_shortcut.weight")
                elif ".shortcut.bias" in new_key:
                    new_key = new_key.replace(".shortcut.bias", ".conv_shortcut.bias")

            # Handle resample blocks - change downsamples to downsampler and remove index
            elif "resample" in new_key or "time_conv" in new_key:
                # Change the second downsamples to downsampler and remove the index
                parts = new_key.split(".")
                # Find the pattern: encoder.down_blocks.X.downsamples.Y.resample...
                # We want to change it to: encoder.down_blocks.X.downsampler.resample...
                if len(parts) >= 4 and parts[3] == "downsamples":
                    # Remove the index (parts[4]) and change downsamples to downsampler
                    new_parts = parts[:3] + ["downsampler"] + parts[5:]
                    new_key = ".".join(new_parts)

            new_state_dict[new_key] = value

        # Handle decoder upsamples
        elif key.startswith("decoder.upsamples."):
            # Change decoder.upsamples to decoder.up_blocks
            new_key = key.replace("decoder.upsamples.", "decoder.up_blocks.")

            # Handle residual blocks - change upsamples to resnets and rename components
            if "residual" in new_key or "shortcut" in new_key:
                # Change the second upsamples to resnets
                new_key = new_key.replace(".upsamples.", ".resnets.")

                # Rename residual components
                if ".residual.0.gamma" in new_key:
                    new_key = new_key.replace(".residual.0.gamma", ".norm1.gamma")
                elif ".residual.2.weight" in new_key:
                    new_key = new_key.replace(".residual.2.weight", ".conv1.weight")
                elif ".residual.2.bias" in new_key:
                    new_key = new_key.replace(".residual.2.bias", ".conv1.bias")
                elif ".residual.3.gamma" in new_key:
                    new_key = new_key.replace(".residual.3.gamma", ".norm2.gamma")
                elif ".residual.6.weight" in new_key:
                    new_key = new_key.replace(".residual.6.weight", ".conv2.weight")
                elif ".residual.6.bias" in new_key:
                    new_key = new_key.replace(".residual.6.bias", ".conv2.bias")
                elif ".shortcut.weight" in new_key:
                    new_key = new_key.replace(".shortcut.weight", ".conv_shortcut.weight")
                elif ".shortcut.bias" in new_key:
                    new_key = new_key.replace(".shortcut.bias", ".conv_shortcut.bias")

            # Handle resample blocks - change upsamples to upsampler and remove index
            elif "resample" in new_key or "time_conv" in new_key:
                # Change the second upsamples to upsampler and remove the index
                parts = new_key.split(".")
                # Find the pattern: encoder.down_blocks.X.downsamples.Y.resample...
                # We want to change it to: encoder.down_blocks.X.downsampler.resample...
                if len(parts) >= 4 and parts[3] == "upsamples":
                    # Remove the index (parts[4]) and change upsamples to upsampler
                    new_parts = parts[:3] + ["upsampler"] + parts[5:]
                    new_key = ".".join(new_parts)

            new_state_dict[new_key] = value
        else:
            # Keep other keys unchanged
            new_state_dict[key] = value

    with init_empty_weights():
        vae = AutoencoderKLWan(**vae22_diffusers_config)
    vae.load_state_dict(new_state_dict, strict=True, assign=True)
    return vae


def convert_openclip_xlm_roberta_vit_to_clip_vision_model():
    """
    Convert OpenCLIP XLM-RoBERTa-CLIP vision encoder to HuggingFace CLIPVisionModel format.

    The original checkpoint contains a multimodal XLM-RoBERTa-CLIP model with:
    - Vision encoder: ViT-Huge/14 (1280 dim, 32 layers, 16 heads, patch_size=14)
    - Text encoder: XLM-RoBERTa-Large (not used in Wan2.2-Animate)

    We extract only the vision encoder and convert it to CLIPVisionModel format.

    IMPORTANT: The original uses use_31_block=True (returns features from first 31 blocks only).
    We convert only the first 31 layers to match this behavior exactly.
    """
    # Download the OpenCLIP checkpoint
    checkpoint_path = hf_hub_download(
        "Wan-AI/Wan2.2-Animate-14B", "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
    )

    # Load the checkpoint
    openclip_state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    # Create mapping from OpenCLIP vision encoder to CLIPVisionModel
    # OpenCLIP uses "visual." prefix, we need to map to CLIPVisionModel structure
    clip_vision_state_dict = {}

    # Mapping rules:
    # visual.patch_embedding.weight -> vision_model.embeddings.patch_embedding.weight
    # visual.patch_embedding.bias -> vision_model.embeddings.patch_embedding.bias
    # visual.cls_embedding -> vision_model.embeddings.class_embedding
    # visual.pos_embedding -> vision_model.embeddings.position_embedding.weight
    # visual.transformer.{i}.norm1.weight -> vision_model.encoder.layers.{i}.layer_norm1.weight
    # visual.transformer.{i}.norm1.bias -> vision_model.encoder.layers.{i}.layer_norm1.bias
    # visual.transformer.{i}.attn.to_qkv.weight -> split into to_q, to_k, to_v
    # visual.transformer.{i}.attn.proj.weight -> vision_model.encoder.layers.{i}.self_attn.out_proj.weight
    # visual.transformer.{i}.norm2.weight -> vision_model.encoder.layers.{i}.layer_norm2.weight
    # visual.transformer.{i}.mlp.0.weight -> vision_model.encoder.layers.{i}.mlp.fc1.weight
    # visual.transformer.{i}.mlp.2.weight -> vision_model.encoder.layers.{i}.mlp.fc2.weight
    # visual.pre_norm -> vision_model.pre_layrnorm (if exists)
    # visual.post_norm -> vision_model.post_layernorm (if exists)

    for key, value in openclip_state_dict.items():
        if not key.startswith("visual."):
            # Skip text encoder and other components
            continue

        # Remove "visual." prefix
        new_key = key[7:]  # Remove "visual."

        # Embeddings
        if new_key == "patch_embedding.weight":
            clip_vision_state_dict["vision_model.embeddings.patch_embedding.weight"] = value
        elif new_key == "patch_embedding.bias":
            clip_vision_state_dict["vision_model.embeddings.patch_embedding.bias"] = value
        elif new_key == "cls_embedding":
            # Remove extra batch dimension: [1, 1, 1280] -> [1280]
            clip_vision_state_dict["vision_model.embeddings.class_embedding"] = value.squeeze()
        elif new_key == "pos_embedding":
            # Remove extra batch dimension: [1, 257, 1280] -> [257, 1280]
            clip_vision_state_dict["vision_model.embeddings.position_embedding.weight"] = value.squeeze(0)

        # Pre-norm (if exists)
        elif new_key == "pre_norm.weight":
            clip_vision_state_dict["vision_model.pre_layrnorm.weight"] = value
        elif new_key == "pre_norm.bias":
            clip_vision_state_dict["vision_model.pre_layrnorm.bias"] = value

        # Post-norm - final layer norm after transformer blocks
        elif new_key == "post_norm.weight":
            clip_vision_state_dict["vision_model.post_layernorm.weight"] = value
        elif new_key == "post_norm.bias":
            clip_vision_state_dict["vision_model.post_layernorm.bias"] = value

        # Transformer layers (only first 31 layers, skip layer 31 which is index 31)
        elif new_key.startswith("transformer."):
            parts = new_key.split(".")
            if len(parts) >= 3:
                layer_idx = int(parts[1])

                # Skip the 32nd layer (index 31) to match use_31_block=True
                if layer_idx >= 31:
                    continue

                component = ".".join(parts[2:])

                # Layer norm 1
                if component == "norm1.weight":
                    clip_vision_state_dict[f"vision_model.encoder.layers.{layer_idx}.layer_norm1.weight"] = value
                elif component == "norm1.bias":
                    clip_vision_state_dict[f"vision_model.encoder.layers.{layer_idx}.layer_norm1.bias"] = value

                # Attention - QKV split
                elif component == "attn.to_qkv.weight":
                    # Split QKV into separate Q, K, V
                    qkv = value
                    q, k, v = qkv.chunk(3, dim=0)
                    clip_vision_state_dict[f"vision_model.encoder.layers.{layer_idx}.self_attn.q_proj.weight"] = q
                    clip_vision_state_dict[f"vision_model.encoder.layers.{layer_idx}.self_attn.k_proj.weight"] = k
                    clip_vision_state_dict[f"vision_model.encoder.layers.{layer_idx}.self_attn.v_proj.weight"] = v
                elif component == "attn.to_qkv.bias":
                    # Split QKV bias
                    qkv_bias = value
                    q_bias, k_bias, v_bias = qkv_bias.chunk(3, dim=0)
                    clip_vision_state_dict[f"vision_model.encoder.layers.{layer_idx}.self_attn.q_proj.bias"] = q_bias
                    clip_vision_state_dict[f"vision_model.encoder.layers.{layer_idx}.self_attn.k_proj.bias"] = k_bias
                    clip_vision_state_dict[f"vision_model.encoder.layers.{layer_idx}.self_attn.v_proj.bias"] = v_bias

                # Attention output projection
                elif component == "attn.proj.weight":
                    clip_vision_state_dict[f"vision_model.encoder.layers.{layer_idx}.self_attn.out_proj.weight"] = (
                        value
                    )
                elif component == "attn.proj.bias":
                    clip_vision_state_dict[f"vision_model.encoder.layers.{layer_idx}.self_attn.out_proj.bias"] = value

                # Layer norm 2
                elif component == "norm2.weight":
                    clip_vision_state_dict[f"vision_model.encoder.layers.{layer_idx}.layer_norm2.weight"] = value
                elif component == "norm2.bias":
                    clip_vision_state_dict[f"vision_model.encoder.layers.{layer_idx}.layer_norm2.bias"] = value

                # MLP
                elif component.startswith("mlp.0."):
                    # First linear layer
                    mlp_component = component[6:]  # Remove "mlp.0."
                    clip_vision_state_dict[f"vision_model.encoder.layers.{layer_idx}.mlp.fc1.{mlp_component}"] = value
                elif component.startswith("mlp.2."):
                    # Second linear layer (after activation)
                    mlp_component = component[6:]  # Remove "mlp.2."
                    clip_vision_state_dict[f"vision_model.encoder.layers.{layer_idx}.mlp.fc2.{mlp_component}"] = value

    # Create CLIPVisionModel with matching config
    # Use 31 layers to match the original use_31_block=True behavior
    config = CLIPVisionConfig(
        hidden_size=1280,
        intermediate_size=5120,  # 1280 * 4 (mlp_ratio)
        num_hidden_layers=31,  # Only first 31 layers, matching use_31_block=True
        num_attention_heads=16,
        image_size=224,
        patch_size=14,
        hidden_act="gelu",
        layer_norm_eps=1e-5,
        attention_dropout=0.0,
        projection_dim=1024,  # embed_dim from original config
    )

    with init_empty_weights():
        vision_model = CLIPVisionModel(config)

    # Load state dict into the meta model, which will materialize the tensors
    vision_model.load_state_dict(clip_vision_state_dict, strict=True, assign=True)

    # Move to CPU to ensure all tensors are materialized
    vision_model = vision_model.to("cpu")

    return vision_model


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default=None)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--dtype", default="fp32", choices=["fp32", "fp16", "bf16", "none"])
    return parser.parse_args()


DTYPE_MAPPING = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


if __name__ == "__main__":
    args = get_args()

    if "Wan2.2" in args.model_type and "TI2V" not in args.model_type and "Animate" not in args.model_type:
        transformer = convert_transformer(args.model_type, stage="high_noise_model")
        transformer_2 = convert_transformer(args.model_type, stage="low_noise_model")
    else:
        transformer = convert_transformer(args.model_type)
        transformer_2 = None

    if "Wan2.2" in args.model_type and "TI2V" in args.model_type:
        vae = convert_vae_22()
    else:
        vae = convert_vae()

    text_encoder = UMT5EncoderModel.from_pretrained("google/umt5-xxl", torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained("google/umt5-xxl")
    if "FLF2V" in args.model_type:
        flow_shift = 16.0
    elif "TI2V" in args.model_type or "Animate" in args.model_type:
        flow_shift = 5.0
    else:
        flow_shift = 3.0
    scheduler = UniPCMultistepScheduler(
        prediction_type="flow_prediction", use_flow_sigmas=True, num_train_timesteps=1000, flow_shift=flow_shift
    )

    # If user has specified "none", we keep the original dtypes of the state dict without any conversion
    if args.dtype != "none":
        dtype = DTYPE_MAPPING[args.dtype]
        transformer.to(dtype)
        if transformer_2 is not None:
            transformer_2.to(dtype)

    if "Wan2.2" and "I2V" in args.model_type and "TI2V" not in args.model_type:
        pipe = WanImageToVideoPipeline(
            transformer=transformer,
            transformer_2=transformer_2,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            vae=vae,
            scheduler=scheduler,
            boundary_ratio=0.9,
        )
    elif "Wan2.2" and "T2V" in args.model_type:
        pipe = WanPipeline(
            transformer=transformer,
            transformer_2=transformer_2,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            vae=vae,
            scheduler=scheduler,
            boundary_ratio=0.875,
        )
    elif "Wan2.2" and "TI2V" in args.model_type:
        pipe = WanPipeline(
            transformer=transformer,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            vae=vae,
            scheduler=scheduler,
            expand_timesteps=True,
        )
    elif "I2V" in args.model_type or "FLF2V" in args.model_type:
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            "laion/CLIP-ViT-H-14-laion2B-s32B-b79K", torch_dtype=torch.bfloat16
        )
        image_processor = AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
        pipe = WanImageToVideoPipeline(
            transformer=transformer,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            vae=vae,
            scheduler=scheduler,
            image_encoder=image_encoder,
            image_processor=image_processor,
        )
    elif "Wan2.2-VACE" in args.model_type:
        pipe = WanVACEPipeline(
            transformer=transformer,
            transformer_2=transformer_2,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            vae=vae,
            scheduler=scheduler,
            boundary_ratio=0.875,
        )
    elif "Wan-VACE" in args.model_type:
        pipe = WanVACEPipeline(
            transformer=transformer,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            vae=vae,
            scheduler=scheduler,
        )
    elif "Animate" in args.model_type:
        # Convert OpenCLIP XLM-RoBERTa-CLIP vision encoder to CLIPVisionModel
        print("Converting XLM-RoBERTa-CLIP vision encoder from OpenCLIP checkpoint...")
        image_encoder = convert_openclip_xlm_roberta_vit_to_clip_vision_model()

        # Create image processor for ViT-Huge/14 with 224x224 images
        image_processor = CLIPImageProcessor(
            size={"shortest_edge": 224},
            crop_size={"height": 224, "width": 224},
            do_center_crop=True,
            do_normalize=True,
            do_rescale=True,
            do_resize=True,
            image_mean=[0.48145466, 0.4578275, 0.40821073],
            image_std=[0.26862954, 0.26130258, 0.27577711],
            resample=3,  # PIL.Image.BICUBIC
            rescale_factor=0.00392156862745098,  # 1/255
        )

        pipe = WanAnimatePipeline(
            transformer=transformer,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            vae=vae,
            scheduler=scheduler,
            image_encoder=image_encoder,
            image_processor=image_processor,
        )
    else:
        pipe = WanPipeline(
            transformer=transformer,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            vae=vae,
            scheduler=scheduler,
        )

    pipe.save_pretrained(args.output_path, safe_serialization=True, max_shard_size="5GB")
