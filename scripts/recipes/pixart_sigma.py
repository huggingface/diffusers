import argparse
import os

import torch
from transformers import T5EncoderModel, T5Tokenizer

from diffusers import AutoencoderKL, DPMSolverMultistepScheduler, PixArtSigmaPipeline, Transformer2DModel
from diffusers.loaders.conversion.checkpoint import convert_component_checkpoint


ckpt_id = "PixArt-alpha"
# https://github.com/PixArt-alpha/PixArt-sigma/blob/dd087141864e30ec44f12cb7448dd654be065e88/scripts/inference.py#L158
interpolation_scale = {256: 0.5, 512: 1, 1024: 2, 2048: 4}


def main(args):
    all_state_dict = torch.load(args.orig_ckpt_path)
    state_dict = all_state_dict.pop("state_dict")
    transformer = Transformer2DModel(
        sample_size=args.image_size // 8,
        num_layers=28,
        attention_head_dim=72,
        in_channels=4,
        out_channels=8,
        patch_size=2,
        attention_bias=True,
        num_attention_heads=16,
        cross_attention_dim=1152,
        activation_fn="gelu-approximate",
        num_embeds_ada_norm=1000,
        norm_type="ada_norm_single",
        norm_elementwise_affine=False,
        norm_eps=1e-6,
        caption_channels=4096,
        interpolation_scale=interpolation_scale[args.image_size],
        use_additional_conditions=args.micro_condition,
    )
    transformer.load_state_dict(
        convert_component_checkpoint(state_dict, dict(transformer.config), "PixArtTransformer2DModel"), strict=True
    )

    num_model_params = sum(p.numel() for p in transformer.parameters())
    print(f"Total number of transformer parameters: {num_model_params}")

    if args.only_transformer:
        transformer.save_pretrained(os.path.join(args.dump_path, "transformer"))
    else:
        # pixart-Sigma vae link: https://huggingface.co/PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers/tree/main/vae
        vae = AutoencoderKL.from_pretrained(f"{ckpt_id}/pixart_sigma_sdxlvae_T5_diffusers", subfolder="vae")

        scheduler = DPMSolverMultistepScheduler()

        tokenizer = T5Tokenizer.from_pretrained(f"{ckpt_id}/pixart_sigma_sdxlvae_T5_diffusers", subfolder="tokenizer")
        text_encoder = T5EncoderModel.from_pretrained(
            f"{ckpt_id}/pixart_sigma_sdxlvae_T5_diffusers", subfolder="text_encoder"
        )

        pipeline = PixArtSigmaPipeline(
            tokenizer=tokenizer, text_encoder=text_encoder, transformer=transformer, vae=vae, scheduler=scheduler
        )

        pipeline.save_pretrained(args.dump_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--micro_condition", action="store_true", help="If use Micro-condition in PixArtMS structure during training."
    )
    parser.add_argument("--qk_norm", action="store_true", help="If use qk norm during training.")
    parser.add_argument(
        "--orig_ckpt_path", default=None, type=str, required=False, help="Path to the checkpoint to convert."
    )
    parser.add_argument(
        "--image_size",
        default=1024,
        type=int,
        choices=[256, 512, 1024, 2048],
        required=False,
        help="Image size of pretrained model, 256, 512, 1024, or 2048.",
    )
    parser.add_argument("--dump_path", default=None, type=str, required=True, help="Path to the output pipeline.")
    parser.add_argument("--only_transformer", default=True, type=bool, required=True)

    args = parser.parse_args()
    main(args)
