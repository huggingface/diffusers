import argparse
import os

import torch
from safetensors.torch import load_file
from transformers import AutoModel, AutoTokenizer

from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, LuminaNextDiT2DModel, LuminaText2ImgPipeline


def main(args):
    # checkpoint from https://huggingface.co/Alpha-VLLM/Lumina-Next-SFT or https://huggingface.co/Alpha-VLLM/Lumina-Next-T2I
    all_sd = load_file(args.origin_ckpt_path, device="cpu")
    converted_state_dict = {}
    # pad token
    converted_state_dict["pad_token"] = all_sd["pad_token"]

    # patch embed
    converted_state_dict["patch_embedder.weight"] = all_sd["x_embedder.weight"]
    converted_state_dict["patch_embedder.bias"] = all_sd["x_embedder.bias"]

    # time and caption embed
    converted_state_dict["time_caption_embed.timestep_embedder.linear_1.weight"] = all_sd["t_embedder.mlp.0.weight"]
    converted_state_dict["time_caption_embed.timestep_embedder.linear_1.bias"] = all_sd["t_embedder.mlp.0.bias"]
    converted_state_dict["time_caption_embed.timestep_embedder.linear_2.weight"] = all_sd["t_embedder.mlp.2.weight"]
    converted_state_dict["time_caption_embed.timestep_embedder.linear_2.bias"] = all_sd["t_embedder.mlp.2.bias"]
    converted_state_dict["time_caption_embed.caption_embedder.0.weight"] = all_sd["cap_embedder.0.weight"]
    converted_state_dict["time_caption_embed.caption_embedder.0.bias"] = all_sd["cap_embedder.0.bias"]
    converted_state_dict["time_caption_embed.caption_embedder.1.weight"] = all_sd["cap_embedder.1.weight"]
    converted_state_dict["time_caption_embed.caption_embedder.1.bias"] = all_sd["cap_embedder.1.bias"]

    for i in range(24):
        # adaln
        converted_state_dict[f"layers.{i}.gate"] = all_sd[f"layers.{i}.attention.gate"]
        converted_state_dict[f"layers.{i}.adaLN_modulation.1.weight"] = all_sd[f"layers.{i}.adaLN_modulation.1.weight"]
        converted_state_dict[f"layers.{i}.adaLN_modulation.1.bias"] = all_sd[f"layers.{i}.adaLN_modulation.1.bias"]

        # qkv
        converted_state_dict[f"layers.{i}.attn1.to_q.weight"] = all_sd[f"layers.{i}.attention.wq.weight"]
        converted_state_dict[f"layers.{i}.attn1.to_k.weight"] = all_sd[f"layers.{i}.attention.wk.weight"]
        converted_state_dict[f"layers.{i}.attn1.to_v.weight"] = all_sd[f"layers.{i}.attention.wv.weight"]

        # cap
        converted_state_dict[f"layers.{i}.attn2.to_q.weight"] = all_sd[f"layers.{i}.attention.wq.weight"]
        converted_state_dict[f"layers.{i}.attn2.to_k.weight"] = all_sd[f"layers.{i}.attention.wk_y.weight"]
        converted_state_dict[f"layers.{i}.attn2.to_v.weight"] = all_sd[f"layers.{i}.attention.wv_y.weight"]

        # output
        converted_state_dict[f"layers.{i}.attn2.to_out.0.weight"] = all_sd[f"layers.{i}.attention.wo.weight"]

        # attention
        # qk norm
        converted_state_dict[f"layers.{i}.attn1.norm_q.weight"] = all_sd[f"layers.{i}.attention.q_norm.weight"]
        converted_state_dict[f"layers.{i}.attn1.norm_q.bias"] = all_sd[f"layers.{i}.attention.q_norm.bias"]

        converted_state_dict[f"layers.{i}.attn1.norm_k.weight"] = all_sd[f"layers.{i}.attention.k_norm.weight"]
        converted_state_dict[f"layers.{i}.attn1.norm_k.bias"] = all_sd[f"layers.{i}.attention.k_norm.bias"]

        converted_state_dict[f"layers.{i}.attn2.norm_q.weight"] = all_sd[f"layers.{i}.attention.q_norm.weight"]
        converted_state_dict[f"layers.{i}.attn2.norm_q.bias"] = all_sd[f"layers.{i}.attention.q_norm.bias"]

        converted_state_dict[f"layers.{i}.attn2.norm_k.weight"] = all_sd[f"layers.{i}.attention.ky_norm.weight"]
        converted_state_dict[f"layers.{i}.attn2.norm_k.bias"] = all_sd[f"layers.{i}.attention.ky_norm.bias"]

        # attention norm
        converted_state_dict[f"layers.{i}.attn_norm1.weight"] = all_sd[f"layers.{i}.attention_norm1.weight"]
        converted_state_dict[f"layers.{i}.attn_norm2.weight"] = all_sd[f"layers.{i}.attention_norm2.weight"]
        converted_state_dict[f"layers.{i}.norm1_context.weight"] = all_sd[f"layers.{i}.attention_y_norm.weight"]

        # feed forward
        converted_state_dict[f"layers.{i}.feed_forward.linear_1.weight"] = all_sd[f"layers.{i}.feed_forward.w1.weight"]
        converted_state_dict[f"layers.{i}.feed_forward.linear_2.weight"] = all_sd[f"layers.{i}.feed_forward.w2.weight"]
        converted_state_dict[f"layers.{i}.feed_forward.linear_3.weight"] = all_sd[f"layers.{i}.feed_forward.w3.weight"]

        # feed forward norm
        converted_state_dict[f"layers.{i}.ffn_norm1.weight"] = all_sd[f"layers.{i}.ffn_norm1.weight"]
        converted_state_dict[f"layers.{i}.ffn_norm2.weight"] = all_sd[f"layers.{i}.ffn_norm2.weight"]

    # final layer
    converted_state_dict["final_layer.linear.weight"] = all_sd["final_layer.linear.weight"]
    converted_state_dict["final_layer.linear.bias"] = all_sd["final_layer.linear.bias"]

    converted_state_dict["final_layer.adaLN_modulation.1.weight"] = all_sd["final_layer.adaLN_modulation.1.weight"]
    converted_state_dict["final_layer.adaLN_modulation.1.bias"] = all_sd["final_layer.adaLN_modulation.1.bias"]

    # Lumina-Next-SFT 2B
    transformer = LuminaNextDiT2DModel(
        sample_size=128,
        patch_size=2,
        in_channels=4,
        hidden_size=2304,
        num_layers=24,
        num_attention_heads=32,
        num_kv_heads=8,
        multiple_of=256,
        ffn_dim_multiplier=None,
        norm_eps=1e-5,
        learn_sigma=True,
        qk_norm=True,
        cross_attention_dim=2048,
        scaling_factor=1.0,
    )
    transformer.load_state_dict(converted_state_dict, strict=True)

    num_model_params = sum(p.numel() for p in transformer.parameters())
    print(f"Total number of transformer parameters: {num_model_params}")

    if args.only_transformer:
        transformer.save_pretrained(os.path.join(args.dump_path, "transformer"))
    else:
        scheduler = FlowMatchEulerDiscreteScheduler()

        vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae", torch_dtype=torch.float32)

        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
        text_encoder = AutoModel.from_pretrained("google/gemma-2b")

        pipeline = LuminaText2ImgPipeline(
            tokenizer=tokenizer, text_encoder=text_encoder, transformer=transformer, vae=vae, scheduler=scheduler
        )
        pipeline.save_pretrained(args.dump_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--origin_ckpt_path", default=None, type=str, required=False, help="Path to the checkpoint to convert."
    )
    parser.add_argument(
        "--image_size",
        default=1024,
        type=int,
        choices=[256, 512, 1024],
        required=False,
        help="Image size of pretrained model, either 512 or 1024.",
    )
    parser.add_argument("--dump_path", default=None, type=str, required=True, help="Path to the output pipeline.")
    parser.add_argument("--only_transformer", default=True, type=bool, required=True)

    args = parser.parse_args()
    main(args)
