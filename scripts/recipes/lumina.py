import argparse
import os

import torch
from safetensors.torch import load_file
from transformers import AutoModel, AutoTokenizer

from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, LuminaNextDiT2DModel, LuminaPipeline
from diffusers.loaders.conversion.checkpoint import convert_component_checkpoint


def main(args):
    # checkpoint from https://huggingface.co/Alpha-VLLM/Lumina-Next-SFT or https://huggingface.co/Alpha-VLLM/Lumina-Next-T2I
    all_sd = load_file(args.origin_ckpt_path, device="cpu")
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
    transformer.load_state_dict(
        convert_component_checkpoint(all_sd, dict(transformer.config), "LuminaNextDiT2DModel"), strict=True
    )

    num_model_params = sum(p.numel() for p in transformer.parameters())
    print(f"Total number of transformer parameters: {num_model_params}")

    if args.only_transformer:
        transformer.save_pretrained(os.path.join(args.dump_path, "transformer"))
    else:
        scheduler = FlowMatchEulerDiscreteScheduler()

        vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae", torch_dtype=torch.float32)

        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
        text_encoder = AutoModel.from_pretrained("google/gemma-2b")

        pipeline = LuminaPipeline(
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
