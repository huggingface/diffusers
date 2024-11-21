#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
from contextlib import nullcontext

import torch
from accelerate import init_empty_weights
from diffusers import (
    DCAE,
    DCAE_HF,
    FlowDPMSolverMultistepScheduler,
    FlowMatchEulerDiscreteScheduler,
    SanaPAGPipeline,
    SanaTransformer2DModel,
)
from diffusers.models.modeling_utils import load_model_dict_into_meta
from diffusers.utils.import_utils import is_accelerate_available
from termcolor import colored
from transformers import AutoModelForCausalLM, AutoTokenizer

CTX = init_empty_weights if is_accelerate_available else nullcontext

ckpt_id = "Sana"
# https://github.com/NVlabs/Sana/blob/main/scripts/inference.py


def main(args):
    all_state_dict = torch.load(args.orig_ckpt_path, map_location=torch.device("cpu"))
    state_dict = all_state_dict.pop("state_dict")
    converted_state_dict = {}

    # Patch embeddings.
    converted_state_dict["pos_embed.proj.weight"] = state_dict.pop("x_embedder.proj.weight")
    converted_state_dict["pos_embed.proj.bias"] = state_dict.pop("x_embedder.proj.bias")

    # Caption projection.
    converted_state_dict["caption_projection.linear_1.weight"] = state_dict.pop("y_embedder.y_proj.fc1.weight")
    converted_state_dict["caption_projection.linear_1.bias"] = state_dict.pop("y_embedder.y_proj.fc1.bias")
    converted_state_dict["caption_projection.linear_2.weight"] = state_dict.pop("y_embedder.y_proj.fc2.weight")
    converted_state_dict["caption_projection.linear_2.bias"] = state_dict.pop("y_embedder.y_proj.fc2.bias")

    # AdaLN-single LN
    converted_state_dict["adaln_single.emb.timestep_embedder.linear_1.weight"] = state_dict.pop(
        "t_embedder.mlp.0.weight"
    )
    converted_state_dict["adaln_single.emb.timestep_embedder.linear_1.bias"] = state_dict.pop("t_embedder.mlp.0.bias")
    converted_state_dict["adaln_single.emb.timestep_embedder.linear_2.weight"] = state_dict.pop(
        "t_embedder.mlp.2.weight"
    )
    converted_state_dict["adaln_single.emb.timestep_embedder.linear_2.bias"] = state_dict.pop("t_embedder.mlp.2.bias")

    # Shared norm.
    converted_state_dict["adaln_single.linear.weight"] = state_dict.pop("t_block.1.weight")
    converted_state_dict["adaln_single.linear.bias"] = state_dict.pop("t_block.1.bias")

    # y norm
    converted_state_dict["caption_norm.weight"] = state_dict.pop("attention_y_norm.weight")

    if args.model_type == "SanaMS_1600M_P1_D20":
        layer_num = 20
        flow_shift = 3.0
    elif args.model_type == "SanaMS_600M_P1_D28":
        layer_num = 28
        flow_shift = 4.0
    else:
        raise ValueError(f"{args.model_type} is not supported.")

    for depth in range(layer_num):
        # Transformer blocks.
        converted_state_dict[f"transformer_blocks.{depth}.scale_shift_table"] = state_dict.pop(
            f"blocks.{depth}.scale_shift_table"
        )
        # Linear Attention is all you need 🤘

        # Self attention.
        q, k, v = torch.chunk(state_dict.pop(f"blocks.{depth}.attn.qkv.weight"), 3, dim=0)
        converted_state_dict[f"transformer_blocks.{depth}.attn1.to_q.weight"] = q
        converted_state_dict[f"transformer_blocks.{depth}.attn1.to_k.weight"] = k
        converted_state_dict[f"transformer_blocks.{depth}.attn1.to_v.weight"] = v
        # Projection.
        converted_state_dict[f"transformer_blocks.{depth}.attn1.to_out.0.weight"] = state_dict.pop(
            f"blocks.{depth}.attn.proj.weight"
        )
        converted_state_dict[f"transformer_blocks.{depth}.attn1.to_out.0.bias"] = state_dict.pop(
            f"blocks.{depth}.attn.proj.bias"
        )

        # Feed-forward.
        converted_state_dict[f"transformer_blocks.{depth}.ff.inverted_conv.conv.weight"] = state_dict.pop(
            f"blocks.{depth}.mlp.inverted_conv.conv.weight"
        )
        converted_state_dict[f"transformer_blocks.{depth}.ff.inverted_conv.conv.bias"] = state_dict.pop(
            f"blocks.{depth}.mlp.inverted_conv.conv.bias"
        )
        converted_state_dict[f"transformer_blocks.{depth}.ff.depth_conv.conv.weight"] = state_dict.pop(
            f"blocks.{depth}.mlp.depth_conv.conv.weight"
        )
        converted_state_dict[f"transformer_blocks.{depth}.ff.depth_conv.conv.bias"] = state_dict.pop(
            f"blocks.{depth}.mlp.depth_conv.conv.bias"
        )
        converted_state_dict[f"transformer_blocks.{depth}.ff.point_conv.conv.weight"] = state_dict.pop(
            f"blocks.{depth}.mlp.point_conv.conv.weight"
        )

        # Cross-attention.
        q = state_dict.pop(f"blocks.{depth}.cross_attn.q_linear.weight")
        q_bias = state_dict.pop(f"blocks.{depth}.cross_attn.q_linear.bias")
        k, v = torch.chunk(state_dict.pop(f"blocks.{depth}.cross_attn.kv_linear.weight"), 2, dim=0)
        k_bias, v_bias = torch.chunk(state_dict.pop(f"blocks.{depth}.cross_attn.kv_linear.bias"), 2, dim=0)

        converted_state_dict[f"transformer_blocks.{depth}.attn2.to_q.weight"] = q
        converted_state_dict[f"transformer_blocks.{depth}.attn2.to_q.bias"] = q_bias
        converted_state_dict[f"transformer_blocks.{depth}.attn2.to_k.weight"] = k
        converted_state_dict[f"transformer_blocks.{depth}.attn2.to_k.bias"] = k_bias
        converted_state_dict[f"transformer_blocks.{depth}.attn2.to_v.weight"] = v
        converted_state_dict[f"transformer_blocks.{depth}.attn2.to_v.bias"] = v_bias

        converted_state_dict[f"transformer_blocks.{depth}.attn2.to_out.0.weight"] = state_dict.pop(
            f"blocks.{depth}.cross_attn.proj.weight"
        )
        converted_state_dict[f"transformer_blocks.{depth}.attn2.to_out.0.bias"] = state_dict.pop(
            f"blocks.{depth}.cross_attn.proj.bias"
        )

    # Final block.
    converted_state_dict["proj_out.weight"] = state_dict.pop("final_layer.linear.weight")
    converted_state_dict["proj_out.bias"] = state_dict.pop("final_layer.linear.bias")
    converted_state_dict["scale_shift_table"] = state_dict.pop("final_layer.scale_shift_table")

    # Transformer
    with CTX():
        transformer = SanaTransformer2DModel(
            num_attention_heads=model_kwargs[args.model_type]["num_attention_heads"],
            attention_head_dim=model_kwargs[args.model_type]["attention_head_dim"],
            num_cross_attention_heads=model_kwargs[args.model_type]["num_cross_attention_heads"],
            cross_attention_head_dim=model_kwargs[args.model_type]["cross_attention_head_dim"],
            in_channels=32,
            out_channels=32,
            num_layers=model_kwargs[args.model_type]["num_layers"],
            cross_attention_dim=model_kwargs[args.model_type]["cross_attention_dim"],
            attention_bias=False,
            sample_size=32,
            patch_size=1,
            activation_fn=("silu", "silu", None),
            upcast_attention=False,
            norm_type="ada_norm_single",
            norm_elementwise_affine=False,
            norm_eps=1e-6,
            use_additional_conditions=False,
            caption_channels=2304,
            use_caption_norm=True,
            caption_norm_scale_factor=0.1,
            attention_type="default",
            use_pe=False,
            expand_ratio=2.5,
            ff_bias=(True, True, False),
            ff_norm=(None, None, None),
        )
    if is_accelerate_available():
        load_model_dict_into_meta(transformer, converted_state_dict)
    else:
        transformer.load_state_dict(converted_state_dict, strict=True)

    try:
        state_dict.pop("y_embedder.y_embedding")
        state_dict.pop("pos_embed")
    except:
        pass
    assert len(state_dict) == 0, f"State dict is not empty, {state_dict.keys()}"

    num_model_params = sum(p.numel() for p in transformer.parameters())
    print(f"Total number of transformer parameters: {num_model_params}")

    if not args.save_full_pipeline:
        print(
            colored(
                f"Only saving transformer model of {args.model_type}. "
                f"Set --save_full_pipeline to save the whole SanaPipeline",
                "green",
                attrs=["bold"],
            )
        )
        transformer.to(weight_dtype).save_pretrained(os.path.join(args.dump_path, "transformer"))
    else:
        print(colored(f"Saving the whole SanaPAGPipeline containing {args.model_type}", "green", attrs=["bold"]))
        # VAE
        dc_ae = DCAE_HF.from_pretrained(f"mit-han-lab/dc-ae-f32c32-sana-1.0")
        dc_ae_state_dict = dc_ae.state_dict()
        dc_ae = DCAE(
            in_channels=3,
            latent_channels=32,
            encoder_width_list=[128, 256, 512, 512, 1024, 1024],
            encoder_depth_list=[2, 2, 2, 3, 3, 3],
            encoder_block_type=["ResBlock", "ResBlock", "ResBlock", "EViTS5_GLU", "EViTS5_GLU", "EViTS5_GLU"],
            encoder_norm="rms2d",
            encoder_act="silu",
            downsample_block_type="Conv",
            decoder_width_list=[128, 256, 512, 512, 1024, 1024],
            decoder_depth_list=[3, 3, 3, 3, 3, 3],
            decoder_block_type=["ResBlock", "ResBlock", "ResBlock", "EViTS5_GLU", "EViTS5_GLU", "EViTS5_GLU"],
            decoder_norm="rms2d",
            decoder_act="silu",
            upsample_block_type="InterpolateConv",
            scaling_factor=0.41407,
        )
        dc_ae.load_state_dict(dc_ae_state_dict, strict=True)
        dc_ae.to(torch.float32).to(device)

        # Text Encoder
        text_encoder_model_path = "google/gemma-2-2b-it"
        tokenizer = AutoTokenizer.from_pretrained(text_encoder_model_path)
        tokenizer.padding_side = "right"
        text_encoder = (
            AutoModelForCausalLM.from_pretrained(text_encoder_model_path, torch_dtype=torch.bfloat16)
            .get_decoder()
            .to(device)
        )

        # Scheduler
        if args.scheduler_type == "flow-dpm_solver":
            scheduler = FlowDPMSolverMultistepScheduler(flow_shift=flow_shift)
        elif args.scheduler_type == "flow-euler":
            scheduler = FlowMatchEulerDiscreteScheduler(shift=flow_shift)
        else:
            raise ValueError(f"Scheduler type {args.scheduler_type} is not supported")

        # transformer
        transformer.to(device).to(weight_dtype)

        pipe = SanaPAGPipeline(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            transformer=transformer,
            vae=dc_ae,
            scheduler=scheduler,
            pag_applied_layers="blocks.8",
        )

        image = pipe(
            "a dog",
            height=1024,
            width=1024,
            guidance_scale=5.0,
            pag_scale=2.0,
        )[0]

        image[0].save("sana_pag.png")

        pipe.save_pretrained(args.dump_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--orig_ckpt_path", default=None, type=str, required=False, help="Path to the checkpoint to convert."
    )
    parser.add_argument(
        "--image_size",
        default=1024,
        type=int,
        choices=[512, 1024],
        required=False,
        help="Image size of pretrained model, 512 or 1024.",
    )
    parser.add_argument(
        "--model_type", default="SanaMS_1600M_P1_D20", type=str, choices=["SanaMS_1600M_P1_D20", "SanaMS_600M_P1_D28"]
    )
    parser.add_argument(
        "--scheduler_type", default="flow-dpm_solver", type=str, choices=["flow-dpm_solver", "flow-euler"]
    )
    parser.add_argument("--dump_path", default=None, type=str, required=True, help="Path to the output pipeline.")
    parser.add_argument("--save_full_pipeline", action="store_true", help="save all the pipelien elemets in one.")

    args = parser.parse_args()

    model_kwargs = {
        "SanaMS_1600M_P1_D20": {
            "num_attention_heads": 70,
            "attention_head_dim": 32,
            "num_cross_attention_heads": 20,
            "cross_attention_head_dim": 112,
            "cross_attention_dim": 2240,
            "num_layers": 20,
        },
        "SanaMS_600M_P1_D28": {
            "num_attention_heads": 36,
            "attention_head_dim": 32,
            "num_cross_attention_heads": 16,
            "cross_attention_head_dim": 72,
            "cross_attention_dim": 1152,
            "num_layers": 28,
        },
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    weight_dtype = torch.float16

    main(args)
