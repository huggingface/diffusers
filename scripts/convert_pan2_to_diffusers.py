"""Convert an original PAN 2 checkpoint into the diffusers layout.

The transformer checkpoint can be a torch distributed checkpoint (DCP) folder, a `model_state.pt` state dict, or a
`.safetensors` file. Every key maps onto a diffusers module by renaming alone, except for:

* the fused QKV projections, which are split into `to_q` / `to_k` / `to_v` (and `add_q_proj` / `add_k_proj` /
  `add_v_proj` for the text stream of the joint-attention blocks),
* the per-modality output projection, feed-forward and q/k norm weights, which are split into the video and the text
  modules,
* the token refiner's SwiGLU `w1` / `w3`, which are stacked into `ff.net.0.proj` as `[w3; w1]` because diffusers'
  `SwiGLU` reads `[value; gate]`,
* the final layer modulation, whose `[shift; scale]` halves are swapped for `AdaLayerNormContinuous`,
* `cond_type_embedding`, of which only the text row (index 0) is used and kept,
* `time_in.freqs`, which is dropped because diffusers recomputes it.

Usage:

```bash
# Transformer only.
python scripts/convert_pan2_to_diffusers.py \
    --checkpoint_path /path/to/checkpoint/transformer --output_path /tmp/pan2-diffusers

# Full modular pipeline repository.
python scripts/convert_pan2_to_diffusers.py \
    --checkpoint_path /path/to/checkpoint/transformer --output_path /tmp/pan2-diffusers --save_pipeline
```
"""

import argparse
import json
import os

import torch
from safetensors.torch import load_file

import diffusers
from diffusers import PAN2Transformer3DModel


PAN2_COMPONENTS = {
    "transformer": ("diffusers", "PAN2Transformer3DModel"),
    "vae": ("diffusers", "AutoencoderKLWan"),
    "scheduler": ("diffusers", "FlowMatchEulerDiscreteScheduler"),
    "text_encoder": ("transformers", "Qwen3_5TextModel"),
    "tokenizer": ("transformers", "Qwen2Tokenizer"),
}

DTYPE_MAPPING = {"fp32": torch.float32, "bf16": torch.bfloat16}


def load_original_state_dict(checkpoint_path: str) -> dict[str, torch.Tensor]:
    if os.path.isdir(checkpoint_path):
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint.metadata import TensorStorageMetadata

        reader = dcp.FileSystemReader(checkpoint_path)
        metadata = reader.read_metadata()
        state_dict = {
            key: torch.empty(value.size, dtype=value.properties.dtype)
            for key, value in metadata.state_dict_metadata.items()
            if isinstance(value, TensorStorageMetadata)
        }
        dcp.load(state_dict, storage_reader=reader, no_dist=True)
    elif checkpoint_path.endswith(".safetensors"):
        state_dict = load_file(checkpoint_path)
    else:
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        for key in ("module", "state_dict", "model"):
            if key in state_dict and isinstance(state_dict[key], dict):
                state_dict = state_dict[key]
    return {key.removeprefix("module.").removeprefix("model."): value for key, value in state_dict.items()}


def convert_pan2_transformer_state_dict(
    original_state_dict: dict[str, torch.Tensor], num_layers: int, num_refiner_layers: int, attention_head_dim: int
) -> dict[str, torch.Tensor]:
    original_state_dict = dict(original_state_dict)
    state_dict = {}

    def move(source_key: str, target_key: str) -> None:
        state_dict[target_key] = original_state_dict.pop(source_key)

    move("img_in.proj.weight", "x_embedder.weight")
    move("img_in.proj.bias", "x_embedder.bias")

    original_state_dict.pop("time_in.freqs")
    move("time_in.mlp.0.weight", "time_embedder.linear_1.weight")
    move("time_in.mlp.0.bias", "time_embedder.linear_1.bias")
    move("time_in.mlp.2.weight", "time_embedder.linear_2.weight")
    move("time_in.mlp.2.bias", "time_embedder.linear_2.bias")

    move("cap_embedder.0.weight", "context_embedder.0.weight")
    move("cap_embedder.1.weight", "context_embedder.1.weight")
    move("cap_embedder.1.bias", "context_embedder.1.bias")

    for i in range(num_refiner_layers):
        source = f"txt_in.individual_token_refiner.blocks.{i}."
        target = f"context_refiner.{i}."
        move(source + "norm1.weight", target + "norm1.weight")
        move(source + "norm2.weight", target + "norm2.weight")
        to_q, to_k, to_v = original_state_dict.pop(source + "self_attn_qkv.weight").chunk(3, dim=0)
        state_dict[target + "attn.to_q.weight"] = to_q
        state_dict[target + "attn.to_k.weight"] = to_k
        state_dict[target + "attn.to_v.weight"] = to_v
        move(source + "self_attn_q_norm.weight", target + "attn.norm_q.weight")
        move(source + "self_attn_k_norm.weight", target + "attn.norm_k.weight")
        move(source + "self_attn_proj.weight", target + "attn.to_out.0.weight")
        w1 = original_state_dict.pop(source + "mlp.w1.weight")
        w3 = original_state_dict.pop(source + "mlp.w3.weight")
        state_dict[target + "ff.net.0.proj.weight"] = torch.cat([w3, w1], dim=0)
        move(source + "mlp.w2.weight", target + "ff.net.2.weight")

    # Only the text-encoder row of the condition type embedding is used.
    state_dict["context_type_embedding"] = original_state_dict.pop("cond_type_embedding.weight")[0]

    for i in range(num_layers):
        source = f"single_blocks.{i}."
        target = f"transformer_blocks.{i}."
        move(source + "modulation.linear.weight", target + "norm1.linear.weight")
        move(source + "modulation.linear.bias", target + "norm1.linear.bias")
        move(source + "text_modulation", target + "scale_shift_table_context")

        # The fused QKV projection stacks `[q; k; v]` of the video stream, then of the text stream.
        qkv_weight = original_state_dict.pop(source + "qkv_proj.weight")
        qkv_bias = original_state_dict.pop(source + "qkv_proj.bias")
        names = ["to_q", "to_k", "to_v", "add_q_proj", "add_k_proj", "add_v_proj"]
        for name, weight, bias in zip(names, qkv_weight.chunk(6, dim=0), qkv_bias.chunk(6, dim=0)):
            state_dict[target + f"attn.{name}.weight"] = weight
            state_dict[target + f"attn.{name}.bias"] = bias

        q_norm = original_state_dict.pop(source + "qkv_proj.q_weight")
        k_norm = original_state_dict.pop(source + "qkv_proj.k_weight")
        state_dict[target + "attn.norm_q.weight"] = q_norm[:attention_head_dim]
        state_dict[target + "attn.norm_added_q.weight"] = q_norm[attention_head_dim:]
        state_dict[target + "attn.norm_k.weight"] = k_norm[:attention_head_dim]
        state_dict[target + "attn.norm_added_k.weight"] = k_norm[attention_head_dim:]

        out_weight = original_state_dict.pop(source + "out_proj.weight")
        out_bias = original_state_dict.pop(source + "out_proj.bias")
        state_dict[target + "attn.to_out.0.weight"], state_dict[target + "attn.to_add_out.weight"] = out_weight.chunk(
            2, dim=0
        )
        state_dict[target + "attn.to_out.0.bias"], state_dict[target + "attn.to_add_out.bias"] = out_bias.chunk(2)

        for modality, ff in (("video", "ff"), ("text", "ff_context")):
            move(source + f"ffn.w_up_{modality}", target + f"{ff}.net.0.proj.weight")
            move(source + f"ffn.b_up_{modality}", target + f"{ff}.net.0.proj.bias")
            move(source + f"ffn.w_down_{modality}", target + f"{ff}.net.2.weight")
            move(source + f"ffn.b_down_{modality}", target + f"{ff}.net.2.bias")

    # The reference final layer chunks its modulation as `[shift; scale]`, `AdaLayerNormContinuous` as `[scale; shift]`.
    for suffix in ("weight", "bias"):
        shift, scale = original_state_dict.pop(f"final_layer.adaLN_modulation.1.{suffix}").chunk(2, dim=0)
        state_dict[f"norm_out.linear.{suffix}"] = torch.cat([scale, shift], dim=0)
    move("final_layer.linear.weight", "proj_out.weight")
    move("final_layer.linear.bias", "proj_out.bias")

    if len(original_state_dict) > 0:
        raise ValueError(f"Unconverted keys left in the original state dict: {sorted(original_state_dict)}")
    return state_dict


def convert_transformer(checkpoint_path: str, dtype: torch.dtype) -> PAN2Transformer3DModel:
    with torch.device("meta"):
        transformer = PAN2Transformer3DModel()
    config = transformer.config

    original_state_dict = load_original_state_dict(checkpoint_path)
    state_dict = convert_pan2_transformer_state_dict(
        original_state_dict,
        num_layers=config.num_layers,
        num_refiner_layers=config.num_refiner_layers,
        attention_head_dim=config.attention_head_dim,
    )
    del original_state_dict

    transformer.load_state_dict(state_dict, strict=True, assign=True)
    for name, param in transformer.named_parameters():
        keep_in_fp32 = any(module in name.split(".") for module in transformer._keep_in_fp32_modules)
        param.data = param.data.to(torch.float32 if keep_in_fp32 else dtype)
    return transformer


def write_modular_model_index(output_path: str, repo_id: str) -> None:
    modular_index = {
        "_class_name": "PAN2ModularPipeline",
        "_diffusers_version": diffusers.__version__,
        "_blocks_class_name": "PAN2Blocks",
    }
    for name, (library, class_name) in PAN2_COMPONENTS.items():
        modular_index[name] = [
            library,
            class_name,
            {
                "type_hint": [library, class_name],
                "pretrained_model_name_or_path": repo_id,
                "subfolder": name,
                "variant": None,
                "revision": None,
            },
        ]
    with open(os.path.join(output_path, "modular_model_index.json"), "w") as f:
        json.dump(modular_index, f, indent=2)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Original transformer checkpoint: a DCP folder, a `model_state.pt` file or a `.safetensors` file.",
    )
    parser.add_argument("--output_path", type=str, required=True, help="Where the diffusers checkpoint is written.")
    parser.add_argument("--dtype", type=str, default="bf16", choices=list(DTYPE_MAPPING))
    parser.add_argument(
        "--save_pipeline",
        action="store_true",
        help="Also write the text encoder, tokenizer, VAE and scheduler as a full modular pipeline repository.",
    )
    parser.add_argument(
        "--text_encoder_path",
        type=str,
        default="Qwen/Qwen3.5-9B",
        help="Qwen3.5 checkpoint to take the text encoder and tokenizer from. Used with `--save_pipeline`.",
    )
    parser.add_argument(
        "--vae_path",
        type=str,
        default="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        help="Diffusers Wan2.2 repository to take the VAE from. Used with `--save_pipeline`.",
    )
    parser.add_argument(
        "--modular_repo_id",
        type=str,
        default=None,
        help="Repository the components of `modular_model_index.json` load from. Defaults to `--output_path`.",
    )
    return parser.parse_args()


def main(args):
    dtype = DTYPE_MAPPING[args.dtype]
    os.makedirs(args.output_path, exist_ok=True)

    transformer = convert_transformer(args.checkpoint_path, dtype)
    transformer.save_pretrained(os.path.join(args.output_path, "transformer"), max_shard_size="5GB")
    del transformer

    if not args.save_pipeline:
        return

    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    from diffusers import AutoencoderKLWan, FlowMatchEulerDiscreteScheduler

    # PAN 2 only reads text hidden states, so only the language model of Qwen3.5 is kept.
    text_encoder = Qwen3_5ForConditionalGeneration.from_pretrained(args.text_encoder_path, dtype=dtype)
    text_encoder.model.language_model.save_pretrained(
        os.path.join(args.output_path, "text_encoder"), max_shard_size="5GB"
    )
    del text_encoder
    tokenizer = AutoTokenizer.from_pretrained(args.text_encoder_path)
    tokenizer.save_pretrained(os.path.join(args.output_path, "tokenizer"))

    vae = AutoencoderKLWan.from_pretrained(args.vae_path, subfolder="vae", dtype=torch.float32)
    vae.save_pretrained(os.path.join(args.output_path, "vae"))

    scheduler = FlowMatchEulerDiscreteScheduler(shift=9.0)
    scheduler.save_pretrained(os.path.join(args.output_path, "scheduler"))

    write_modular_model_index(args.output_path, args.modular_repo_id or args.output_path)


if __name__ == "__main__":
    main(get_args())
