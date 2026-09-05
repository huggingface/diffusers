import argparse
from typing import Any, Dict

import torch


def get_state_dict(saved_dict: Dict[str, Any]) -> dict[str, Any]:
    if "state_dict" in saved_dict:
        return saved_dict["state_dict"]

    if "model" in saved_dict:
        return saved_dict["model"]

    return saved_dict


def rename_layernorm_key(key: str) -> str:
    if ".attn1.norm_q." in key:
        return key.replace(
            ".attn1.norm_q.",
            ".attention.query_layernorm_list.0.",
        )

    if ".attn1.norm_k." in key:
        return key.replace(
            ".attn1.norm_k.",
            ".attention.key_layernorm_list.0.",
        )

    return key


def rename_transformer_key(key: str) -> str:
    replacements = [
        ("transformer_blocks", "transformer.layers"),
        (".attn1.to_out.0.", ".attention.dense."),
        (".ff.net.0.proj.", ".mlp.dense_h_to_4h."),
        (".ff.net.2.", ".mlp.dense_4h_to_h."),
        ("norm_final.", "transformer.final_layernorm."),
        ("time_embedding.linear_1.", "time_embed.0."),
        ("time_embedding.linear_2.", "time_embed.2."),
        ("ofs_embedding.linear_1.", "ofs_embed.0."),
        ("ofs_embedding.linear_2.", "ofs_embed.2."),
        ("patch_embed.", "mixins.patch_embed."),
        ("norm_out.norm.", "mixins.final_layer.norm_final."),
        ("norm_out.linear.", "mixins.final_layer.adaLN_modulation.1."),
        ("proj_out.", "mixins.final_layer.linear."),
    ]

    for old, new in replacements:
        key = key.replace(old, new)

    return key


def convert_transformer_state_dict(
    state_dict: Dict[str, Any],
) -> Dict[str, Any]:
    converted_state_dict = {}

    # Reconstruct QKV.
    for key in state_dict:
        if key.endswith(("attn1.to_q.weight", "attn1.to_q.bias")):
            to_k_key = key.replace("to_q", "to_k")
            to_v_key = key.replace("to_q", "to_v")

            to_q = state_dict[key]
            to_k = state_dict[to_k_key]
            to_v = state_dict[to_v_key]

            new_key = rename_transformer_key(
                key.replace(
                    ".attn1.to_q.",
                    ".attention.query_key_value.",
                )
            )

            converted_state_dict[new_key] = torch.cat(
                [to_q, to_k, to_v],
                dim=0,
            )

    # Copy ordinary keys.
    qkv_parts = (
        ".attn1.to_q.",
        ".attn1.to_k.",
        ".attn1.to_v.",
    )

    for key, value in state_dict.items():
        if any(part in key for part in qkv_parts):
            continue

        if ".attn1.norm_q." in key or ".attn1.norm_k." in key:
          new_key = rename_layernorm_key(key)
          new_key = rename_transformer_key(new_key)
          converted_state_dict[new_key] = value
          continue

        if ".norm1.linear." in key or ".norm2.linear." in key:
            continue

        new_key = rename_transformer_key(key)
        converted_state_dict[new_key] = value

    # Reconstruct AdaLN.
    for layer_id in _get_layer_ids(state_dict):
        norm1_prefix = (
            f"transformer_blocks.{layer_id}.norm1.linear."
        )
        norm2_prefix = (
            f"transformer_blocks.{layer_id}.norm2.linear."
        )

        for weight_or_bias in ("weight", "bias"):
            norm1 = state_dict[
                norm1_prefix + weight_or_bias
            ]
            norm2 = state_dict[
                norm2_prefix + weight_or_bias
            ]

            chunks1 = norm1.chunk(6, dim=0)
            chunks2 = norm2.chunk(6, dim=0)

            modulation = torch.cat(
                [
                    chunks1[0],
                    chunks1[1],
                    chunks1[2],
                    chunks2[0],
                    chunks2[1],
                    chunks2[2],
                    chunks1[3],
                    chunks1[4],
                    chunks1[5],
                    chunks2[3],
                    chunks2[4],
                    chunks2[5],
                ],
                dim=0,
            )

            converted_state_dict[
                "transformer.layers."
                f"{layer_id}.adaln_layer."
                f"adaLN_modulations.{weight_or_bias}"
            ] = modulation

    return converted_state_dict


def _get_layer_ids(
    state_dict: Dict[str, Any],
) -> list[str]:
    layer_ids = set()
    prefix = "transformer_blocks."

    for key in state_dict:
        if key.startswith(prefix):
            layer_ids.add(
                key[len(prefix):].split(".", 1)[0]
            )

    return sorted(layer_ids, key=int)
def load_checkpoint(input_path: str) -> Dict[str, Any]:
    print(f"Loading checkpoint from: {input_path}")

    checkpoint = torch.load(
        input_path,
        map_location="cpu",
    )

    state_dict = get_state_dict(checkpoint)

    print(f"Loaded {len(state_dict)} tensors")

    return state_dict
def get_args():
    parser = argparse.ArgumentParser(
        description="Convert a Diffusers CogVideoX transformer checkpoint."
    )

    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to the input Diffusers checkpoint.",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the converted checkpoint.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    print(f"Input:  {args.input_path}")
    print(f"Output: {args.output_path}")

    state_dict = load_checkpoint(args.input_path)

    converted = convert_transformer_state_dict(state_dict)

    print(f"Converted {len(converted)} tensors")

    for key, value in converted.items():
        print(key, tuple(value.shape))