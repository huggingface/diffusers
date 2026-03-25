#!/usr/bin/env python
# coding=utf-8

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from ul_models import ULTwoStageBaseModel

from diffusers import UNet2DModel
from diffusers.training_utils import (
    ul_alpha_sigma_from_logsnr,
    ul_logsnr_schedule,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Sample UL Stage-2 latents and decode to images.")
    parser.add_argument("--stage2_base_path", type=str, required=True, help="Path to stage2 base_model directory.")
    parser.add_argument("--stage1_decoder_path", type=str, required=True, help="Path to stage1 decoder directory.")
    parser.add_argument("--output_path", type=str, required=True, help="Output image path.")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--latent_channels", type=int, default=4)
    parser.add_argument("--latent_downsample_factor", type=int, default=8)
    parser.add_argument("--num_train_timesteps", type=int, default=1000)
    parser.add_argument("--num_sampling_steps", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--base_schedule", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--decoder_schedule", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--lambda_z_min", type=float, default=-10.0)
    parser.add_argument("--lambda_z_max", type=float, default=5.0)
    parser.add_argument("--lambda_x_min", type=float, default=-10.0)
    parser.add_argument("--lambda_x_max", type=float, default=10.0)
    parser.add_argument("--decoder_prediction_type", type=str, default="x0", choices=["epsilon", "x0"])
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    latent_size = args.resolution // args.latent_downsample_factor

    base_model = ULTwoStageBaseModel.from_pretrained(args.stage2_base_path).to(device).eval()
    decoder = UNet2DModel.from_pretrained(args.stage1_decoder_path).to(device).eval()

    # Stage-2 latent sampling (DDIM-like deterministic update with v-pred model).
    z_t = torch.randn(args.batch_size, args.latent_channels, latent_size, latent_size, device=device)
    t_grid = torch.linspace(1.0, 0.0, args.num_sampling_steps + 1, device=device)

    for i in range(args.num_sampling_steps):
        t_cur = t_grid[i].repeat(args.batch_size)
        t_nxt = t_grid[i + 1].repeat(args.batch_size)

        lambda_cur = ul_logsnr_schedule(
            t_cur,
            schedule_type=args.base_schedule,
            lambda_min=args.lambda_z_min,
            lambda_max=args.lambda_z_max,
        )
        lambda_nxt = ul_logsnr_schedule(
            t_nxt,
            schedule_type=args.base_schedule,
            lambda_min=args.lambda_z_min,
            lambda_max=args.lambda_z_max,
        )
        alpha_cur, sigma_cur = ul_alpha_sigma_from_logsnr(lambda_cur)
        alpha_nxt, sigma_nxt = ul_alpha_sigma_from_logsnr(lambda_nxt)

        timestep_idx = (t_cur * (args.num_train_timesteps - 1)).long().clamp(0, args.num_train_timesteps - 1)
        dummy_labels = torch.zeros((args.batch_size,), device=device, dtype=torch.long)
        v_pred = base_model(z_t, timestep_idx, dummy_labels)

        z0_hat = alpha_cur[:, None, None, None] * z_t - sigma_cur[:, None, None, None] * v_pred
        eps_hat = sigma_cur[:, None, None, None] * z_t + alpha_cur[:, None, None, None] * v_pred

        z_t = alpha_nxt[:, None, None, None] * z0_hat + sigma_nxt[:, None, None, None] * eps_hat

    # Paper Sec. 3.3: hand off the noisy latent at logsnr_0 to the decoder.
    z_handoff = z_t

    # Decoder sampling conditioned on final stage-2 latent.
    z0_up = F.interpolate(z_handoff, size=(args.resolution, args.resolution), mode="bilinear", align_corners=False)

    x_t = torch.randn(args.batch_size, 3, args.resolution, args.resolution, device=device)
    x_grid = torch.linspace(1.0, 0.0, args.num_sampling_steps + 1, device=device)

    for i in range(args.num_sampling_steps):
        t_cur = x_grid[i].repeat(args.batch_size)
        t_nxt = x_grid[i + 1].repeat(args.batch_size)

        lambda_cur = ul_logsnr_schedule(
            t_cur,
            schedule_type=args.decoder_schedule,
            lambda_min=args.lambda_x_min,
            lambda_max=args.lambda_x_max,
        )
        lambda_nxt = ul_logsnr_schedule(
            t_nxt,
            schedule_type=args.decoder_schedule,
            lambda_min=args.lambda_x_min,
            lambda_max=args.lambda_x_max,
        )
        alpha_cur, sigma_cur = ul_alpha_sigma_from_logsnr(lambda_cur)
        alpha_nxt, sigma_nxt = ul_alpha_sigma_from_logsnr(lambda_nxt)

        timestep_idx = (t_cur * (args.num_train_timesteps - 1)).long().clamp(0, args.num_train_timesteps - 1)
        decoder_input = torch.cat([x_t, z0_up], dim=1)
        decoder_pred = decoder(decoder_input, timestep_idx).sample
        if args.decoder_prediction_type == "epsilon":
            eps_hat = decoder_pred
            x0_hat = (x_t - sigma_cur[:, None, None, None] * eps_hat) / alpha_cur[:, None, None, None].clamp_min(1e-5)
        else:
            x0_hat = decoder_pred
            eps_hat = (x_t - alpha_cur[:, None, None, None] * x0_hat) / sigma_cur[:, None, None, None].clamp_min(1e-5)
        x_t = alpha_nxt[:, None, None, None] * x0_hat + sigma_nxt[:, None, None, None] * eps_hat

    x_final = x_t.clamp(-1, 1)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_image((x_final + 1.0) / 2.0, output_path, nrow=2)
    print(f"Saved samples to {output_path}")


if __name__ == "__main__":
    main()
