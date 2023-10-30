import argparse
import os

import torch
from transformers import T5EncoderModel, T5Tokenizer

from diffusers import AutoencoderKL, DPMSolverMultistepScheduler, PixArtAlphaPipeline, Transformer2DModel


ckpt_id = "PixArt-alpha/PixArt-alpha"
pretrained_models = {512: "", 1024: "PixArt-XL-2-1024x1024.pth"}


def main(args):
    ckpt = pretrained_models[args.image_size]
    final_path = os.path.join("/home/sayak/PixArt-alpha/scripts", "pretrained_models", ckpt)
    state_dict = torch.load(final_path, map_location=lambda storage, loc: storage)

    # Patch embeddings.
    state_dict["pos_embed.proj.weight"] = state_dict["x_embedder.proj.weight"]
    state_dict["pos_embed.proj.bias"] = state_dict["x_embedder.proj.bias"]
    state_dict.pop("x_embedder.proj.weight")
    state_dict.pop("x_embedder.proj.bias")

    # Caption projection.
    state_dict["caption_projection.y_embedding"] = state_dict["y_embedder.y_embedding"]
    state_dict["caption_projection.mlp.0.weight"] = state_dict["y_embedder.y_proj.fc1.weight"]
    state_dict["caption_projection.mlp.0.bias"] = state_dict["y_embedder.y_proj.fc1.bias"]
    state_dict["caption_projection.mlp.2.weight"] = state_dict["y_embedder.y_proj.fc2.weight"]
    state_dict["caption_projection.mlp.2.bias"] = state_dict["y_embedder.y_proj.fc2.bias"]

    state_dict.pop("y_embedder.y_embedding")
    state_dict.pop("y_embedder.y_proj.fc1.weight")
    state_dict.pop("y_embedder.y_proj.fc1.bias")
    state_dict.pop("y_embedder.y_proj.fc2.weight")
    state_dict.pop("y_embedder.y_proj.fc2.bias")

    # AdaLN-single LN
    state_dict["adaln_single.emb.timestep_embedder.linear_1.weight"] = state_dict["t_embedder.mlp.0.weight"]
    state_dict["adaln_single.emb.timestep_embedder.linear_1.bias"] = state_dict["t_embedder.mlp.0.bias"]
    state_dict["adaln_single.emb.timestep_embedder.linear_2.weight"] = state_dict["t_embedder.mlp.2.weight"]
    state_dict["adaln_single.emb.timestep_embedder.linear_2.bias"] = state_dict["t_embedder.mlp.2.bias"]

    # Resolution.
    state_dict["adaln_single.emb.resolution_embedder.mlp.0.weight"] = state_dict["csize_embedder.mlp.0.weight"]
    state_dict["adaln_single.emb.resolution_embedder.mlp.0.bias"] = state_dict["csize_embedder.mlp.0.bias"]
    state_dict["adaln_single.emb.resolution_embedder.mlp.2.weight"] = state_dict["csize_embedder.mlp.2.weight"]
    state_dict["adaln_single.emb.resolution_embedder.mlp.2.bias"] = state_dict["csize_embedder.mlp.2.bias"]
    # Aspect ratio.
    state_dict["adaln_single.emb.aspect_ratio_embedder.mlp.0.weight"] = state_dict["csize_embedder.mlp.0.weight"]
    state_dict["adaln_single.emb.aspect_ratio_embedder.mlp.0.bias"] = state_dict["csize_embedder.mlp.0.bias"]
    state_dict["adaln_single.emb.aspect_ratio_embedder.mlp.2.weight"] = state_dict["csize_embedder.mlp.2.weight"]
    state_dict["adaln_single.emb.aspect_ratio_embedder.mlp.2.bias"] = state_dict["csize_embedder.mlp.2.bias"]
    # Shared norm.
    state_dict["adaln_single.linear.weight"] = state_dict["t_block.1.weight"]
    state_dict["adaln_single.linear.bias"] = state_dict["t_block.1.bias"]

    state_dict.pop("t_embedder.mlp.0.weight")
    state_dict.pop("t_embedder.mlp.0.bias")
    state_dict.pop("t_embedder.mlp.2.weight")
    state_dict.pop("t_embedder.mlp.2.bias")

    state_dict.pop("csize_embedder.mlp.0.weight")
    state_dict.pop("csize_embedder.mlp.0.bias")
    state_dict.pop("csize_embedder.mlp.2.weight")
    state_dict.pop("csize_embedder.mlp.2.bias")

    state_dict.pop("ar_embedder.mlp.0.weight")
    state_dict.pop("ar_embedder.mlp.0.bias")
    state_dict.pop("ar_embedder.mlp.2.weight")
    state_dict.pop("ar_embedder.mlp.2.bias")

    state_dict.pop("t_block.1.weight")
    state_dict.pop("t_block.1.bias")

    for depth in range(28):
        # Transformer blocks.
        state_dict[f"transformer_blocks.{depth}.scale_shift_table"] = state_dict[f"blocks.{depth}.scale_shift_table"]

        q, k, v = torch.chunk(state_dict[f"blocks.{depth}.attn.qkv.weight"], 3, dim=0)
        q_bias, k_bias, v_bias = torch.chunk(state_dict[f"blocks.{depth}.attn.qkv.bias"], 3, dim=0)

        # Attention is all you need 🤘
        state_dict[f"transformer_blocks.{depth}.attn1.to_q.weight"] = q
        state_dict[f"transformer_blocks.{depth}.attn1.to_q.bias"] = q_bias
        state_dict[f"transformer_blocks.{depth}.attn1.to_k.weight"] = k
        state_dict[f"transformer_blocks.{depth}.attn1.to_k.bias"] = k_bias
        state_dict[f"transformer_blocks.{depth}.attn1.to_v.weight"] = v
        state_dict[f"transformer_blocks.{depth}.attn1.to_v.bias"] = v_bias

        state_dict[f"transformer_blocks.{depth}.attn1.to_out.0.weight"] = state_dict[
            f"blocks.{depth}.attn.proj.weight"
        ]
        state_dict[f"transformer_blocks.{depth}.attn1.to_out.0.bias"] = state_dict[f"blocks.{depth}.attn.proj.bias"]

        state_dict[f"transformer_blocks.{depth}.ff.net.0.proj.weight"] = state_dict[f"blocks.{depth}.mlp.fc1.weight"]
        state_dict[f"transformer_blocks.{depth}.ff.net.0.proj.bias"] = state_dict[f"blocks.{depth}.mlp.fc1.bias"]
        state_dict[f"transformer_blocks.{depth}.ff.net.2.weight"] = state_dict[f"blocks.{depth}.mlp.fc2.weight"]
        state_dict[f"transformer_blocks.{depth}.ff.net.2.bias"] = state_dict[f"blocks.{depth}.mlp.fc2.bias"]

        state_dict.pop(f"blocks.{depth}.attn.qkv.weight")
        state_dict.pop(f"blocks.{depth}.attn.qkv.bias")
        state_dict.pop(f"blocks.{depth}.attn.proj.weight")
        state_dict.pop(f"blocks.{depth}.attn.proj.bias")
        state_dict.pop(f"blocks.{depth}.mlp.fc1.weight")
        state_dict.pop(f"blocks.{depth}.mlp.fc1.bias")
        state_dict.pop(f"blocks.{depth}.mlp.fc2.weight")
        state_dict.pop(f"blocks.{depth}.mlp.fc2.bias")
        state_dict.pop(f"blocks.{depth}.scale_shift_table")

    # Final block.
    state_dict["proj_out.weight"] = state_dict["final_layer.linear.weight"]
    state_dict["proj_out.bias"] = state_dict["final_layer.linear.bias"]
    state_dict["scale_shift_table"] = state_dict["final_layer.scale_shift_table"]

    state_dict.pop("final_layer.linear.weight")
    state_dict.pop("final_layer.linear.bias")
    state_dict.pop("final_layer.scale_shift_table")

    # DiT XL/2
    transformer = Transformer2DModel(
        sample_size=args.image_size // 8,
        num_layers=28,
        attention_head_dim=72,
        in_channels=4,
        out_channels=8,
        patch_size=2,
        attention_bias=True,
        num_attention_heads=16,
        activation_fn="gelu-approximate",
        num_embeds_ada_norm=1000,
        norm_type="ada_norm_single",
        norm_elementwise_affine=False,
        output_type="pixart_dit",
        caption_channels=4096,
    )
    # transformer.load_state_dict(state_dict, strict=True)
    missing, unexpected = transformer.load_state_dict(state_dict, strict=False)

    # log information on the stuff that are yet to be implemented.
    print(f"Missing keys:\n {missing}")
    print(f"Unexpected keys:\n {unexpected}")

    # To be configured.
    scheduler = DPMSolverMultistepScheduler()

    vae = AutoencoderKL.from_pretrained(ckpt_id, subfolder="sd-vae-ft-ema")

    tokenizer = T5Tokenizer.from_pretrained(ckpt_id, subfolder="t5-v1_1-xxl")
    text_encoder = T5EncoderModel.from_pretrained(ckpt_id, subfolder="t5-v1_1-xxl")

    pipeline = PixArtAlphaPipeline(
        tokenizer=tokenizer, text_encoder=text_encoder, transformer=transformer, vae=vae, scheduler=scheduler
    )

    if args.save:
        pipeline.save_pretrained(args.checkpoint_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image_size",
        default=1024,
        type=int,
        choices=[512, 1024],
        required=False,
        help="Image size of pretrained model, either 256 or 512.",
    )
    parser.add_argument(
        "--save", default=True, type=bool, required=False, help="Whether to save the converted pipeline or not."
    )
    parser.add_argument(
        "--checkpoint_path", default=None, type=str, required=True, help="Path to the output pipeline."
    )

    args = parser.parse_args()
    main(args)
