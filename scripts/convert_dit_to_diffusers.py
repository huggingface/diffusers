import argparse
import os

import torch

from diffusers import AutoencoderKL, DDIMScheduler, DiT, DiTPipeline
from torchvision.datasets.utils import download_url


pretrained_models = {512: "DiT-XL-2-512x512.pt", 256: "DiT-XL-2-256x256.pt"}


def download_model(model_name):
    """
    Downloads a pre-trained DiT model from the web.
    """
    local_path = f"pretrained_models/{model_name}"
    if not os.path.isfile(local_path):
        os.makedirs("pretrained_models", exist_ok=True)
        web_path = f"https://dl.fbaipublicfiles.com/DiT/models/{model_name}"
        download_url(web_path, "pretrained_models")
    model = torch.load(local_path, map_location=lambda storage, loc: storage)
    return model


def main(args):
    state_dict = download_model(pretrained_models[args.image_size])
    vae = AutoencoderKL.from_pretrained(args.vae_model)

    state_dict["timestep_embedder.linear_1.weight"] = state_dict["t_embedder.mlp.0.weight"]
    state_dict["timestep_embedder.linear_1.bias"] = state_dict["t_embedder.mlp.0.bias"]
    state_dict["timestep_embedder.linear_2.weight"] = state_dict["t_embedder.mlp.2.weight"]
    state_dict["timestep_embedder.linear_2.bias"] = state_dict["t_embedder.mlp.2.bias"]
    state_dict.pop("t_embedder.mlp.0.weight")
    state_dict.pop("t_embedder.mlp.0.bias")
    state_dict.pop("t_embedder.mlp.2.weight")
    state_dict.pop("t_embedder.mlp.2.bias")

    state_dict["sample_embedder.proj.weight"] = state_dict["x_embedder.proj.weight"]
    state_dict["sample_embedder.proj.bias"] = state_dict["x_embedder.proj.bias"]
    state_dict.pop("x_embedder.proj.weight")
    state_dict.pop("x_embedder.proj.bias")

    state_dict["class_embedder.embedding_table.weight"] = state_dict["y_embedder.embedding_table.weight"]
    state_dict.pop("y_embedder.embedding_table.weight")

    for depth in range(28):
        q, k, v = torch.chunk(state_dict[f"blocks.{depth}.attn.qkv.weight"], 3, dim=0)
        q_bias, k_bias, v_bias = torch.chunk(state_dict[f"blocks.{depth}.attn.qkv.bias"], 3, dim=0)

        state_dict[f"blocks.{depth}.attn.to_q.weight"] = q
        state_dict[f"blocks.{depth}.attn.to_q.bias"] = q_bias
        state_dict[f"blocks.{depth}.attn.to_k.weight"] = k
        state_dict[f"blocks.{depth}.attn.to_k.bias"] = k_bias
        state_dict[f"blocks.{depth}.attn.to_v.weight"] = v
        state_dict[f"blocks.{depth}.attn.to_v.bias"] = v_bias

        state_dict[f"blocks.{depth}.attn.to_out.0.weight"] = state_dict[f"blocks.{depth}.attn.proj.weight"]
        state_dict[f"blocks.{depth}.attn.to_out.0.bias"] = state_dict[f"blocks.{depth}.attn.proj.bias"]

        state_dict.pop(f"blocks.{depth}.attn.qkv.weight")
        state_dict.pop(f"blocks.{depth}.attn.qkv.bias")
        state_dict.pop(f"blocks.{depth}.attn.proj.weight")
        state_dict.pop(f"blocks.{depth}.attn.proj.bias")

    dit = DiT(
        input_size=args.image_size // 8,
        depth=28,
        hidden_size=1152,
        patch_size=2,
        num_heads=16,
    )
    dit.load_state_dict(state_dict)

    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_schedule="linear",
        prediction_type="epsilon",
        clip_sample=False,
    )

    pipeline = DiTPipeline(dit=dit, vae=vae, scheduler=scheduler)

    if args.save:
        pipeline.save_pretrained(args.checkpoint_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image_size",
        default=256,
        type=int,
        required=False,
        help="Image size of pretrained model, either 256 or 512.",
    )
    parser.add_argument(
        "--vae_model",
        default="stabilityai/sd-vae-ft-ema",
        type=str,
        required=False,
        help="Path to pretrained VAE model, either stabilityai/sd-vae-ft-mse or stabilityai/sd-vae-ft-ema.",
    )
    parser.add_argument(
        "--save", default=True, type=bool, required=False, help="Whether to save the converted pipeline or not."
    )
    parser.add_argument(
        "--checkpoint_path", default=None, type=str, required=True, help="Path to the output pipeline."
    )

    args = parser.parse_args()
    main(args)
