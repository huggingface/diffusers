import argparse
import os

import torch
from safetensors.torch import load_file
from transformers import AutoModel, AutoTokenizer, AutoConfig

from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, OmniGenTransformer2DModel, OmniGenPipeline


def main(args):
    # checkpoint from https://huggingface.co/Shitao/OmniGen-v1
    ckpt = load_file(args.origin_ckpt_path, device="cpu")

    mapping_dict = {
        "pos_embed": "patch_embedding.pos_embed",
        "x_embedder.proj.weight": "patch_embedding.output_image_proj.weight",
        "x_embedder.proj.bias": "patch_embedding.output_image_proj.bias",
        "input_x_embedder.proj.weight": "patch_embedding.input_image_proj.weight",
        "input_x_embedder.proj.bias": "patch_embedding.input_image_proj.bias",
        "final_layer.adaLN_modulation.1.weight": "norm_out.linear.weight",
        "final_layer.adaLN_modulation.1.bias": "norm_out.linear.bias",
        "final_layer.linear.weight": "proj_out.weight",
        "final_layer.linear.bias": "proj_out.bias",

    }

    converted_state_dict = {}
    for k, v in ckpt.items():
        # new_ckpt[k] = v
        if k in mapping_dict:
            converted_state_dict[mapping_dict[k]] = v
        else:
            converted_state_dict[k] = v

    transformer_config = AutoConfig.from_pretrained(args.origin_ckpt_path)

    # Lumina-Next-SFT 2B
    transformer = OmniGenTransformer2DModel(
        transformer_config=transformer_config,
        patch_size=2,
        in_channels=4,
        pos_embed_max_size=192,
    )
    transformer.load_state_dict(converted_state_dict, strict=True)

    num_model_params = sum(p.numel() for p in transformer.parameters())
    print(f"Total number of transformer parameters: {num_model_params}")

    scheduler = FlowMatchEulerDiscreteScheduler()

    vae = AutoencoderKL.from_pretrained(args.origin_ckpt_path, torch_dtype=torch.float32)

    tokenizer = AutoTokenizer.from_pretrained(args.origin_ckpt_path)


    pipeline = OmniGenPipeline(
        tokenizer=tokenizer, transformer=transformer, vae=vae, scheduler=scheduler
    )
    pipeline.save_pretrained(args.dump_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--origin_ckpt_path", default=None, type=str, required=False, help="Path to the checkpoint to convert."
    )

    parser.add_argument("--dump_path", default=None, type=str, required=True, help="Path to the output pipeline.")

    args = parser.parse_args()
    main(args)
