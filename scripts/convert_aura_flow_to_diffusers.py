import argparse

import torch
from huggingface_hub import hf_hub_download

from diffusers.models.transformers.auraflow_transformer_2d import AuraFlowTransformer2DModel


def load_original_state_dict(args):
    model_pt = hf_hub_download(repo_id=args.original_state_dict_repo_id, filename="aura_diffusion_pytorch_model.bin")
    state_dict = torch.load(model_pt, map_location="cpu")
    return state_dict


def calculate_layers(state_dict_keys, key_prefix):
    dit_layers = set()
    for k in state_dict_keys:
        if key_prefix in k:
            dit_layers.add(int(k.split(".")[2]))
    print(f"{key_prefix}: {len(dit_layers)}")
    return len(dit_layers)


# similar to SD3 but only for the last norm layer
def swap_scale_shift(weight, dim):
    shift, scale = weight.chunk(2, dim=0)
    new_weight = torch.cat([scale, shift], dim=0)
    return new_weight


def convert_transformer(state_dict):
    converted_state_dict = {}
    state_dict_keys = list(state_dict.keys())

    converted_state_dict["register_tokens"] = state_dict.pop("model.register_tokens")
    converted_state_dict["pos_embed.pos_embed"] = state_dict.pop("model.positional_encoding")
    converted_state_dict["pos_embed.proj.weight"] = state_dict.pop("model.init_x_linear.weight")
    converted_state_dict["pos_embed.proj.bias"] = state_dict.pop("model.init_x_linear.bias")

    converted_state_dict["time_step_proj.linear_1.weight"] = state_dict.pop("model.t_embedder.mlp.0.weight")
    converted_state_dict["time_step_proj.linear_1.bias"] = state_dict.pop("model.t_embedder.mlp.0.bias")
    converted_state_dict["time_step_proj.linear_2.weight"] = state_dict.pop("model.t_embedder.mlp.2.weight")
    converted_state_dict["time_step_proj.linear_2.bias"] = state_dict.pop("model.t_embedder.mlp.2.bias")

    converted_state_dict["context_embedder.weight"] = state_dict.pop("model.cond_seq_linear.weight")

    mmdit_layers = calculate_layers(state_dict_keys, key_prefix="double_layers")
    single_dit_layers = calculate_layers(state_dict_keys, key_prefix="single_layers")

    # MMDiT blocks 🎸.
    for i in range(mmdit_layers):
        # feed-forward
        path_mapping = {"mlpX": "ff", "mlpC": "ff_context"}
        weight_mapping = {"c_fc1": "linear_1", "c_fc2": "linear_2", "c_proj": "out_projection"}
        for orig_k, diffuser_k in path_mapping.items():
            for k, v in weight_mapping.items():
                converted_state_dict[f"joint_transformer_blocks.{i}.{diffuser_k}.{v}.weight"] = state_dict.pop(
                    f"model.double_layers.{i}.{orig_k}.{k}.weight"
                )

        # norms
        path_mapping = {"modX": "norm1", "modC": "norm1_context"}
        for orig_k, diffuser_k in path_mapping.items():
            converted_state_dict[f"joint_transformer_blocks.{i}.{diffuser_k}.linear.weight"] = state_dict.pop(
                f"model.double_layers.{i}.{orig_k}.1.weight"
            )

        # attns
        x_attn_mapping = {"w2q": "to_q", "w2k": "to_k", "w2v": "to_v", "w2o": "to_out.0"}
        context_attn_mapping = {"w1q": "add_q_proj", "w1k": "add_k_proj", "w1v": "add_v_proj", "w1o": "to_add_out"}
        for attn_mapping in [x_attn_mapping, context_attn_mapping]:
            for k, v in attn_mapping.items():
                converted_state_dict[f"joint_transformer_blocks.{i}.attn.{v}.weight"] = state_dict.pop(
                    f"model.double_layers.{i}.attn.{k}.weight"
                )

    # Single-DiT blocks.
    for i in range(single_dit_layers):
        # feed-forward
        mapping = {"c_fc1": "linear_1", "c_fc2": "linear_2", "c_proj": "out_projection"}
        for k, v in mapping.items():
            converted_state_dict[f"single_transformer_blocks.{i}.ff.{v}.weight"] = state_dict.pop(
                f"model.single_layers.{i}.mlp.{k}.weight"
            )

        # norms
        converted_state_dict[f"single_transformer_blocks.{i}.norm1.linear.weight"] = state_dict.pop(
            f"model.single_layers.{i}.modCX.1.weight"
        )

        # attns
        x_attn_mapping = {"w1q": "to_q", "w1k": "to_k", "w1v": "to_v", "w1o": "to_out.0"}
        for k, v in x_attn_mapping.items():
            converted_state_dict[f"single_transformer_blocks.{i}.attn.{v}.weight"] = state_dict.pop(
                f"model.single_layers.{i}.attn.{k}.weight"
            )

    # Final blocks.
    converted_state_dict["proj_out.weight"] = state_dict.pop("model.final_linear.weight")
    converted_state_dict["norm_out.linear.weight"] = swap_scale_shift(state_dict.pop("model.modF.1.weight"), dim=None)

    return converted_state_dict


@torch.no_grad()
def populate_state_dict(args):
    original_state_dict = load_original_state_dict(args)
    state_dict_keys = list(original_state_dict.keys())
    mmdit_layers = calculate_layers(state_dict_keys, key_prefix="double_layers")
    single_dit_layers = calculate_layers(state_dict_keys, key_prefix="single_layers")

    converted_state_dict = convert_transformer(original_state_dict)
    model_diffusers = AuraFlowTransformer2DModel(
        num_mmdit_layers=mmdit_layers, num_single_dit_layers=single_dit_layers
    )
    model_diffusers.load_state_dict(converted_state_dict, strict=True)

    return model_diffusers


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_state_dict_repo_id", default="AuraDiffusion/auradiffusion-v0.1a0", type=str)
    parser.add_argument("--dump_path", default="aura-flow", type=str)
    parser.add_argument("--hub_id", default=None, type=str)
    args = parser.parse_args()

    model_diffusers = populate_state_dict(args)
    model_diffusers.save_pretrained(args.dump_path)
    if args.hub_id is not None:
        model_diffusers.push_to_hub(args.hub_id)
