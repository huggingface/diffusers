import argparse
from typing import Any, Dict

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from diffusers import AutoencoderDC


def remap_qkv_(key: str, state_dict: Dict[str, Any]):
    qkv = state_dict.pop(key)
    q, k, v = torch.chunk(qkv, 3, dim=0)
    parent_module, _, _ = key.rpartition(".qkv.conv.weight")
    state_dict[f"{parent_module}.to_q.weight"] = q.squeeze()
    state_dict[f"{parent_module}.to_k.weight"] = k.squeeze()
    state_dict[f"{parent_module}.to_v.weight"] = v.squeeze()


def remap_proj_conv_(key: str, state_dict: Dict[str, Any]):
    parent_module, _, _ = key.rpartition(".proj.conv.weight")
    state_dict[f"{parent_module}.to_out.weight"] = state_dict.pop(key).squeeze()


AE_KEYS_RENAME_DICT = {
    # common
    "main.": "",
    "op_list.": "",
    "context_module": "attn",
    "local_module": "conv_out",
    # NOTE: The below two lines work because scales in the available configs only have a tuple length of 1
    # If there were more scales, there would be more layers, so a loop would be better to handle this
    "aggreg.0.0": "to_qkv_multiscale.0.proj_in",
    "aggreg.0.1": "to_qkv_multiscale.0.proj_out",
    "depth_conv.conv": "conv_depth",
    "inverted_conv.conv": "conv_inverted",
    "point_conv.conv": "conv_point",
    "point_conv.norm": "norm",
    "conv.conv.": "conv.",
    "conv1.conv": "conv1",
    "conv2.conv": "conv2",
    "conv2.norm": "norm",
    "proj.norm": "norm_out",
    # encoder
    "encoder.project_in.conv": "encoder.conv_in",
    "encoder.project_out.0.conv": "encoder.conv_out",
    "encoder.stages": "encoder.down_blocks",
    # decoder
    "decoder.project_in.conv": "decoder.conv_in",
    "decoder.project_out.0": "decoder.norm_out",
    "decoder.project_out.2.conv": "decoder.conv_out",
    "decoder.stages": "decoder.up_blocks",
}

AE_F32C32_KEYS = {
    # encoder
    "encoder.project_in.conv": "encoder.conv_in.conv",
    # decoder
    "decoder.project_out.2.conv": "decoder.conv_out.conv",
}

AE_F64C128_KEYS = {
    # encoder
    "encoder.project_in.conv": "encoder.conv_in.conv",
    # decoder
    "decoder.project_out.2.conv": "decoder.conv_out.conv",
}

AE_F128C512_KEYS = {
    # encoder
    "encoder.project_in.conv": "encoder.conv_in.conv",
    # decoder
    "decoder.project_out.2.conv": "decoder.conv_out.conv",
}

AE_SPECIAL_KEYS_REMAP = {
    "qkv.conv.weight": remap_qkv_,
    "proj.conv.weight": remap_proj_conv_,
}


def get_state_dict(saved_dict: Dict[str, Any]) -> Dict[str, Any]:
    state_dict = saved_dict
    if "model" in saved_dict.keys():
        state_dict = state_dict["model"]
    if "module" in saved_dict.keys():
        state_dict = state_dict["module"]
    if "state_dict" in saved_dict.keys():
        state_dict = state_dict["state_dict"]
    return state_dict


def update_state_dict_(state_dict: Dict[str, Any], old_key: str, new_key: str) -> Dict[str, Any]:
    state_dict[new_key] = state_dict.pop(old_key)


def convert_ae(config_name: str, dtype: torch.dtype):
    config = get_ae_config(config_name)
    hub_id = f"mit-han-lab/{config_name}"
    ckpt_path = hf_hub_download(hub_id, "model.safetensors")
    original_state_dict = get_state_dict(load_file(ckpt_path))

    ae = AutoencoderDC(**config).to(dtype=dtype)

    for key in list(original_state_dict.keys()):
        new_key = key[:]
        for replace_key, rename_key in AE_KEYS_RENAME_DICT.items():
            new_key = new_key.replace(replace_key, rename_key)
        update_state_dict_(original_state_dict, key, new_key)

    for key in list(original_state_dict.keys()):
        for special_key, handler_fn_inplace in AE_SPECIAL_KEYS_REMAP.items():
            if special_key not in key:
                continue
            handler_fn_inplace(key, original_state_dict)

    ae.load_state_dict(original_state_dict, strict=True)
    return ae


def get_ae_config(name: str):
    if name in ["dc-ae-f32c32-sana-1.0"]:
        config = {
            "latent_channels": 32,
            "encoder_block_types": (
                "ResBlock",
                "ResBlock",
                "ResBlock",
                "EfficientViTBlock",
                "EfficientViTBlock",
                "EfficientViTBlock",
            ),
            "decoder_block_types": (
                "ResBlock",
                "ResBlock",
                "ResBlock",
                "EfficientViTBlock",
                "EfficientViTBlock",
                "EfficientViTBlock",
            ),
            "encoder_block_out_channels": (128, 256, 512, 512, 1024, 1024),
            "decoder_block_out_channels": (128, 256, 512, 512, 1024, 1024),
            "encoder_qkv_multiscales": ((), (), (), (5,), (5,), (5,)),
            "decoder_qkv_multiscales": ((), (), (), (5,), (5,), (5,)),
            "encoder_layers_per_block": (2, 2, 2, 3, 3, 3),
            "decoder_layers_per_block": [3, 3, 3, 3, 3, 3],
            "downsample_block_type": "conv",
            "upsample_block_type": "interpolate",
            "decoder_norm_types": "rms_norm",
            "decoder_act_fns": "silu",
            "scaling_factor": 0.41407,
        }
    elif name in ["dc-ae-f32c32-in-1.0", "dc-ae-f32c32-mix-1.0"]:
        AE_KEYS_RENAME_DICT.update(AE_F32C32_KEYS)
        config = {
            "latent_channels": 32,
            "encoder_block_types": [
                "ResBlock",
                "ResBlock",
                "ResBlock",
                "EfficientViTBlock",
                "EfficientViTBlock",
                "EfficientViTBlock",
            ],
            "decoder_block_types": [
                "ResBlock",
                "ResBlock",
                "ResBlock",
                "EfficientViTBlock",
                "EfficientViTBlock",
                "EfficientViTBlock",
            ],
            "encoder_block_out_channels": [128, 256, 512, 512, 1024, 1024],
            "decoder_block_out_channels": [128, 256, 512, 512, 1024, 1024],
            "encoder_layers_per_block": [0, 4, 8, 2, 2, 2],
            "decoder_layers_per_block": [0, 5, 10, 2, 2, 2],
            "encoder_qkv_multiscales": ((), (), (), (), (), ()),
            "decoder_qkv_multiscales": ((), (), (), (), (), ()),
            "decoder_norm_types": ["batch_norm", "batch_norm", "batch_norm", "rms_norm", "rms_norm", "rms_norm"],
            "decoder_act_fns": ["relu", "relu", "relu", "silu", "silu", "silu"],
        }
        if name == "dc-ae-f32c32-in-1.0":
            config["scaling_factor"] = 0.3189
        elif name == "dc-ae-f32c32-mix-1.0":
            config["scaling_factor"] = 0.4552
    elif name in ["dc-ae-f64c128-in-1.0", "dc-ae-f64c128-mix-1.0"]:
        AE_KEYS_RENAME_DICT.update(AE_F64C128_KEYS)
        config = {
            "latent_channels": 128,
            "encoder_block_types": [
                "ResBlock",
                "ResBlock",
                "ResBlock",
                "EfficientViTBlock",
                "EfficientViTBlock",
                "EfficientViTBlock",
                "EfficientViTBlock",
            ],
            "decoder_block_types": [
                "ResBlock",
                "ResBlock",
                "ResBlock",
                "EfficientViTBlock",
                "EfficientViTBlock",
                "EfficientViTBlock",
                "EfficientViTBlock",
            ],
            "encoder_block_out_channels": [128, 256, 512, 512, 1024, 1024, 2048],
            "decoder_block_out_channels": [128, 256, 512, 512, 1024, 1024, 2048],
            "encoder_layers_per_block": [0, 4, 8, 2, 2, 2, 2],
            "decoder_layers_per_block": [0, 5, 10, 2, 2, 2, 2],
            "encoder_qkv_multiscales": ((), (), (), (), (), (), ()),
            "decoder_qkv_multiscales": ((), (), (), (), (), (), ()),
            "decoder_norm_types": [
                "batch_norm",
                "batch_norm",
                "batch_norm",
                "rms_norm",
                "rms_norm",
                "rms_norm",
                "rms_norm",
            ],
            "decoder_act_fns": ["relu", "relu", "relu", "silu", "silu", "silu", "silu"],
        }
        if name == "dc-ae-f64c128-in-1.0":
            config["scaling_factor"] = 0.2889
        elif name == "dc-ae-f64c128-mix-1.0":
            config["scaling_factor"] = 0.4538
    elif name in ["dc-ae-f128c512-in-1.0", "dc-ae-f128c512-mix-1.0"]:
        AE_KEYS_RENAME_DICT.update(AE_F128C512_KEYS)
        config = {
            "latent_channels": 512,
            "encoder_block_types": [
                "ResBlock",
                "ResBlock",
                "ResBlock",
                "EfficientViTBlock",
                "EfficientViTBlock",
                "EfficientViTBlock",
                "EfficientViTBlock",
                "EfficientViTBlock",
            ],
            "decoder_block_types": [
                "ResBlock",
                "ResBlock",
                "ResBlock",
                "EfficientViTBlock",
                "EfficientViTBlock",
                "EfficientViTBlock",
                "EfficientViTBlock",
                "EfficientViTBlock",
            ],
            "encoder_block_out_channels": [128, 256, 512, 512, 1024, 1024, 2048, 2048],
            "decoder_block_out_channels": [128, 256, 512, 512, 1024, 1024, 2048, 2048],
            "encoder_layers_per_block": [0, 4, 8, 2, 2, 2, 2, 2],
            "decoder_layers_per_block": [0, 5, 10, 2, 2, 2, 2, 2],
            "encoder_qkv_multiscales": ((), (), (), (), (), (), (), ()),
            "decoder_qkv_multiscales": ((), (), (), (), (), (), (), ()),
            "decoder_norm_types": [
                "batch_norm",
                "batch_norm",
                "batch_norm",
                "rms_norm",
                "rms_norm",
                "rms_norm",
                "rms_norm",
                "rms_norm",
            ],
            "decoder_act_fns": ["relu", "relu", "relu", "silu", "silu", "silu", "silu", "silu"],
        }
        if name == "dc-ae-f128c512-in-1.0":
            config["scaling_factor"] = 0.4883
        elif name == "dc-ae-f128c512-mix-1.0":
            config["scaling_factor"] = 0.3620
    else:
        raise ValueError("Invalid config name provided.")

    return config


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_name",
        type=str,
        default="dc-ae-f32c32-sana-1.0",
        choices=[
            "dc-ae-f32c32-sana-1.0",
            "dc-ae-f32c32-in-1.0",
            "dc-ae-f32c32-mix-1.0",
            "dc-ae-f64c128-in-1.0",
            "dc-ae-f64c128-mix-1.0",
            "dc-ae-f128c512-in-1.0",
            "dc-ae-f128c512-mix-1.0",
        ],
        help="The DCAE checkpoint to convert",
    )
    parser.add_argument("--output_path", type=str, required=True, help="Path where converted model should be saved")
    parser.add_argument("--dtype", default="fp32", help="Torch dtype to save the model in.")
    return parser.parse_args()


DTYPE_MAPPING = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

VARIANT_MAPPING = {
    "fp32": None,
    "fp16": "fp16",
    "bf16": "bf16",
}


if __name__ == "__main__":
    args = get_args()

    dtype = DTYPE_MAPPING[args.dtype]
    variant = VARIANT_MAPPING[args.dtype]

    ae = convert_ae(args.config_name, dtype)
    ae.save_pretrained(args.output_path, safe_serialization=True, max_shard_size="5GB", variant=variant)
