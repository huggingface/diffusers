import argparse

import torch

from diffusers import HunyuanDiT2DControlNetModel


def main(args):
    state_dict = torch.load(args.pt_checkpoint_path, map_location="cpu")

    if args.load_key != "none":
        try:
            state_dict = state_dict[args.load_key]
        except KeyError:
            raise KeyError(
                f"{args.load_key} not found in the checkpoint."
                "Please load from the following keys:{state_dict.keys()}"
            )
    device = "cuda"

    model_config = HunyuanDiT2DControlNetModel.load_config(
        "Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers", subfolder="transformer"
    )
    model_config[
        "use_style_cond_and_image_meta_size"
    ] = args.use_style_cond_and_image_meta_size  ### version <= v1.1: True; version >= v1.2: False
    print(model_config)

    for key in state_dict:
        print("local:", key)

    model = HunyuanDiT2DControlNetModel.from_config(model_config).to(device)

    for key in model.state_dict():
        print("diffusers:", key)

    num_layers = 19
    for i in range(num_layers):
        # attn1
        # Wkqv -> to_q, to_k, to_v
        q, k, v = torch.chunk(state_dict[f"blocks.{i}.attn1.Wqkv.weight"], 3, dim=0)
        q_bias, k_bias, v_bias = torch.chunk(state_dict[f"blocks.{i}.attn1.Wqkv.bias"], 3, dim=0)
        state_dict[f"blocks.{i}.attn1.to_q.weight"] = q
        state_dict[f"blocks.{i}.attn1.to_q.bias"] = q_bias
        state_dict[f"blocks.{i}.attn1.to_k.weight"] = k
        state_dict[f"blocks.{i}.attn1.to_k.bias"] = k_bias
        state_dict[f"blocks.{i}.attn1.to_v.weight"] = v
        state_dict[f"blocks.{i}.attn1.to_v.bias"] = v_bias
        state_dict.pop(f"blocks.{i}.attn1.Wqkv.weight")
        state_dict.pop(f"blocks.{i}.attn1.Wqkv.bias")

        # q_norm, k_norm -> norm_q, norm_k
        state_dict[f"blocks.{i}.attn1.norm_q.weight"] = state_dict[f"blocks.{i}.attn1.q_norm.weight"]
        state_dict[f"blocks.{i}.attn1.norm_q.bias"] = state_dict[f"blocks.{i}.attn1.q_norm.bias"]
        state_dict[f"blocks.{i}.attn1.norm_k.weight"] = state_dict[f"blocks.{i}.attn1.k_norm.weight"]
        state_dict[f"blocks.{i}.attn1.norm_k.bias"] = state_dict[f"blocks.{i}.attn1.k_norm.bias"]

        state_dict.pop(f"blocks.{i}.attn1.q_norm.weight")
        state_dict.pop(f"blocks.{i}.attn1.q_norm.bias")
        state_dict.pop(f"blocks.{i}.attn1.k_norm.weight")
        state_dict.pop(f"blocks.{i}.attn1.k_norm.bias")

        # out_proj -> to_out
        state_dict[f"blocks.{i}.attn1.to_out.0.weight"] = state_dict[f"blocks.{i}.attn1.out_proj.weight"]
        state_dict[f"blocks.{i}.attn1.to_out.0.bias"] = state_dict[f"blocks.{i}.attn1.out_proj.bias"]
        state_dict.pop(f"blocks.{i}.attn1.out_proj.weight")
        state_dict.pop(f"blocks.{i}.attn1.out_proj.bias")

        # attn2
        # kq_proj -> to_k, to_v
        k, v = torch.chunk(state_dict[f"blocks.{i}.attn2.kv_proj.weight"], 2, dim=0)
        k_bias, v_bias = torch.chunk(state_dict[f"blocks.{i}.attn2.kv_proj.bias"], 2, dim=0)
        state_dict[f"blocks.{i}.attn2.to_k.weight"] = k
        state_dict[f"blocks.{i}.attn2.to_k.bias"] = k_bias
        state_dict[f"blocks.{i}.attn2.to_v.weight"] = v
        state_dict[f"blocks.{i}.attn2.to_v.bias"] = v_bias
        state_dict.pop(f"blocks.{i}.attn2.kv_proj.weight")
        state_dict.pop(f"blocks.{i}.attn2.kv_proj.bias")

        # q_proj -> to_q
        state_dict[f"blocks.{i}.attn2.to_q.weight"] = state_dict[f"blocks.{i}.attn2.q_proj.weight"]
        state_dict[f"blocks.{i}.attn2.to_q.bias"] = state_dict[f"blocks.{i}.attn2.q_proj.bias"]
        state_dict.pop(f"blocks.{i}.attn2.q_proj.weight")
        state_dict.pop(f"blocks.{i}.attn2.q_proj.bias")

        # q_norm, k_norm -> norm_q, norm_k
        state_dict[f"blocks.{i}.attn2.norm_q.weight"] = state_dict[f"blocks.{i}.attn2.q_norm.weight"]
        state_dict[f"blocks.{i}.attn2.norm_q.bias"] = state_dict[f"blocks.{i}.attn2.q_norm.bias"]
        state_dict[f"blocks.{i}.attn2.norm_k.weight"] = state_dict[f"blocks.{i}.attn2.k_norm.weight"]
        state_dict[f"blocks.{i}.attn2.norm_k.bias"] = state_dict[f"blocks.{i}.attn2.k_norm.bias"]

        state_dict.pop(f"blocks.{i}.attn2.q_norm.weight")
        state_dict.pop(f"blocks.{i}.attn2.q_norm.bias")
        state_dict.pop(f"blocks.{i}.attn2.k_norm.weight")
        state_dict.pop(f"blocks.{i}.attn2.k_norm.bias")

        # out_proj -> to_out
        state_dict[f"blocks.{i}.attn2.to_out.0.weight"] = state_dict[f"blocks.{i}.attn2.out_proj.weight"]
        state_dict[f"blocks.{i}.attn2.to_out.0.bias"] = state_dict[f"blocks.{i}.attn2.out_proj.bias"]
        state_dict.pop(f"blocks.{i}.attn2.out_proj.weight")
        state_dict.pop(f"blocks.{i}.attn2.out_proj.bias")

        # switch norm 2 and norm 3
        norm2_weight = state_dict[f"blocks.{i}.norm2.weight"]
        norm2_bias = state_dict[f"blocks.{i}.norm2.bias"]
        state_dict[f"blocks.{i}.norm2.weight"] = state_dict[f"blocks.{i}.norm3.weight"]
        state_dict[f"blocks.{i}.norm2.bias"] = state_dict[f"blocks.{i}.norm3.bias"]
        state_dict[f"blocks.{i}.norm3.weight"] = norm2_weight
        state_dict[f"blocks.{i}.norm3.bias"] = norm2_bias

        # norm1 -> norm1.norm
        # default_modulation.1 -> norm1.linear
        state_dict[f"blocks.{i}.norm1.norm.weight"] = state_dict[f"blocks.{i}.norm1.weight"]
        state_dict[f"blocks.{i}.norm1.norm.bias"] = state_dict[f"blocks.{i}.norm1.bias"]
        state_dict[f"blocks.{i}.norm1.linear.weight"] = state_dict[f"blocks.{i}.default_modulation.1.weight"]
        state_dict[f"blocks.{i}.norm1.linear.bias"] = state_dict[f"blocks.{i}.default_modulation.1.bias"]
        state_dict.pop(f"blocks.{i}.norm1.weight")
        state_dict.pop(f"blocks.{i}.norm1.bias")
        state_dict.pop(f"blocks.{i}.default_modulation.1.weight")
        state_dict.pop(f"blocks.{i}.default_modulation.1.bias")

        # mlp.fc1 -> ff.net.0, mlp.fc2 -> ff.net.2
        state_dict[f"blocks.{i}.ff.net.0.proj.weight"] = state_dict[f"blocks.{i}.mlp.fc1.weight"]
        state_dict[f"blocks.{i}.ff.net.0.proj.bias"] = state_dict[f"blocks.{i}.mlp.fc1.bias"]
        state_dict[f"blocks.{i}.ff.net.2.weight"] = state_dict[f"blocks.{i}.mlp.fc2.weight"]
        state_dict[f"blocks.{i}.ff.net.2.bias"] = state_dict[f"blocks.{i}.mlp.fc2.bias"]
        state_dict.pop(f"blocks.{i}.mlp.fc1.weight")
        state_dict.pop(f"blocks.{i}.mlp.fc1.bias")
        state_dict.pop(f"blocks.{i}.mlp.fc2.weight")
        state_dict.pop(f"blocks.{i}.mlp.fc2.bias")

        # after_proj_list -> controlnet_blocks
        state_dict[f"controlnet_blocks.{i}.weight"] = state_dict[f"after_proj_list.{i}.weight"]
        state_dict[f"controlnet_blocks.{i}.bias"] = state_dict[f"after_proj_list.{i}.bias"]
        state_dict.pop(f"after_proj_list.{i}.weight")
        state_dict.pop(f"after_proj_list.{i}.bias")

    # before_proj -> input_block
    state_dict["input_block.weight"] = state_dict["before_proj.weight"]
    state_dict["input_block.bias"] = state_dict["before_proj.bias"]
    state_dict.pop("before_proj.weight")
    state_dict.pop("before_proj.bias")

    # pooler -> time_extra_emb
    state_dict["time_extra_emb.pooler.positional_embedding"] = state_dict["pooler.positional_embedding"]
    state_dict["time_extra_emb.pooler.k_proj.weight"] = state_dict["pooler.k_proj.weight"]
    state_dict["time_extra_emb.pooler.k_proj.bias"] = state_dict["pooler.k_proj.bias"]
    state_dict["time_extra_emb.pooler.q_proj.weight"] = state_dict["pooler.q_proj.weight"]
    state_dict["time_extra_emb.pooler.q_proj.bias"] = state_dict["pooler.q_proj.bias"]
    state_dict["time_extra_emb.pooler.v_proj.weight"] = state_dict["pooler.v_proj.weight"]
    state_dict["time_extra_emb.pooler.v_proj.bias"] = state_dict["pooler.v_proj.bias"]
    state_dict["time_extra_emb.pooler.c_proj.weight"] = state_dict["pooler.c_proj.weight"]
    state_dict["time_extra_emb.pooler.c_proj.bias"] = state_dict["pooler.c_proj.bias"]
    state_dict.pop("pooler.k_proj.weight")
    state_dict.pop("pooler.k_proj.bias")
    state_dict.pop("pooler.q_proj.weight")
    state_dict.pop("pooler.q_proj.bias")
    state_dict.pop("pooler.v_proj.weight")
    state_dict.pop("pooler.v_proj.bias")
    state_dict.pop("pooler.c_proj.weight")
    state_dict.pop("pooler.c_proj.bias")
    state_dict.pop("pooler.positional_embedding")

    # t_embedder -> time_embedding (`TimestepEmbedding`)
    state_dict["time_extra_emb.timestep_embedder.linear_1.bias"] = state_dict["t_embedder.mlp.0.bias"]
    state_dict["time_extra_emb.timestep_embedder.linear_1.weight"] = state_dict["t_embedder.mlp.0.weight"]
    state_dict["time_extra_emb.timestep_embedder.linear_2.bias"] = state_dict["t_embedder.mlp.2.bias"]
    state_dict["time_extra_emb.timestep_embedder.linear_2.weight"] = state_dict["t_embedder.mlp.2.weight"]

    state_dict.pop("t_embedder.mlp.0.bias")
    state_dict.pop("t_embedder.mlp.0.weight")
    state_dict.pop("t_embedder.mlp.2.bias")
    state_dict.pop("t_embedder.mlp.2.weight")

    # x_embedder -> pos_embd (`PatchEmbed`)
    state_dict["pos_embed.proj.weight"] = state_dict["x_embedder.proj.weight"]
    state_dict["pos_embed.proj.bias"] = state_dict["x_embedder.proj.bias"]
    state_dict.pop("x_embedder.proj.weight")
    state_dict.pop("x_embedder.proj.bias")

    # mlp_t5 -> text_embedder
    state_dict["text_embedder.linear_1.bias"] = state_dict["mlp_t5.0.bias"]
    state_dict["text_embedder.linear_1.weight"] = state_dict["mlp_t5.0.weight"]
    state_dict["text_embedder.linear_2.bias"] = state_dict["mlp_t5.2.bias"]
    state_dict["text_embedder.linear_2.weight"] = state_dict["mlp_t5.2.weight"]
    state_dict.pop("mlp_t5.0.bias")
    state_dict.pop("mlp_t5.0.weight")
    state_dict.pop("mlp_t5.2.bias")
    state_dict.pop("mlp_t5.2.weight")

    # extra_embedder -> extra_embedder
    state_dict["time_extra_emb.extra_embedder.linear_1.bias"] = state_dict["extra_embedder.0.bias"]
    state_dict["time_extra_emb.extra_embedder.linear_1.weight"] = state_dict["extra_embedder.0.weight"]
    state_dict["time_extra_emb.extra_embedder.linear_2.bias"] = state_dict["extra_embedder.2.bias"]
    state_dict["time_extra_emb.extra_embedder.linear_2.weight"] = state_dict["extra_embedder.2.weight"]
    state_dict.pop("extra_embedder.0.bias")
    state_dict.pop("extra_embedder.0.weight")
    state_dict.pop("extra_embedder.2.bias")
    state_dict.pop("extra_embedder.2.weight")

    # style_embedder
    if model_config["use_style_cond_and_image_meta_size"]:
        print(state_dict["style_embedder.weight"])
        print(state_dict["style_embedder.weight"].shape)
        state_dict["time_extra_emb.style_embedder.weight"] = state_dict["style_embedder.weight"][0:1]
        state_dict.pop("style_embedder.weight")

    model.load_state_dict(state_dict)

    if args.save:
        model.save_pretrained(args.output_checkpoint_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--save", default=True, type=bool, required=False, help="Whether to save the converted pipeline or not."
    )
    parser.add_argument(
        "--pt_checkpoint_path", default=None, type=str, required=True, help="Path to the .pt pretrained model."
    )
    parser.add_argument(
        "--output_checkpoint_path",
        default=None,
        type=str,
        required=False,
        help="Path to the output converted diffusers pipeline.",
    )
    parser.add_argument(
        "--load_key", default="none", type=str, required=False, help="The key to load from the pretrained .pt file"
    )
    parser.add_argument(
        "--use_style_cond_and_image_meta_size",
        type=bool,
        default=False,
        help="version <= v1.1: True; version >= v1.2: False",
    )

    args = parser.parse_args()
    main(args)
