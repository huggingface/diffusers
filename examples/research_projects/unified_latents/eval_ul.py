#!/usr/bin/env python
# coding=utf-8

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import Inception_V3_Weights, inception_v3
from torchvision.utils import make_grid, save_image
from ul_models import ULTwoStageBaseModel

from diffusers import UNet2DModel
from diffusers.models.autoencoders import AutoencoderULEncoder
from diffusers.training_utils import ul_alpha_sigma_from_logsnr, ul_logsnr_schedule


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Unified Latents stage-1 reconstruction and stage-2 realism."
    )
    parser.add_argument("--train_data_dir", type=str, required=True, help="ImageFolder root with real images.")
    parser.add_argument("--stage1_encoder_path", type=str, required=True, help="Path to stage-1 encoder directory.")
    parser.add_argument("--stage1_decoder_path", type=str, required=True, help="Path to stage-1 decoder directory.")
    parser.add_argument("--stage2_base_path", type=str, required=True, help="Path to stage-2 base model directory.")
    parser.add_argument("--output_dir", type=str, default="ul-eval", help="Directory to store evaluation outputs.")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--num_recon_samples", type=int, default=256)
    parser.add_argument("--num_gen_samples", type=int, default=1024)
    parser.add_argument("--num_sampling_steps", type=int, default=30)
    parser.add_argument("--num_train_timesteps", type=int, default=1000)
    parser.add_argument("--latent_channels", type=int, default=4)
    parser.add_argument("--latent_downsample_factor", type=int, default=8)

    parser.add_argument("--base_schedule", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--decoder_schedule", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--lambda_z_min", type=float, default=-10.0)
    parser.add_argument("--lambda_z_max", type=float, default=5.0)
    parser.add_argument("--lambda_x_min", type=float, default=-10.0)
    parser.add_argument("--lambda_x_max", type=float, default=10.0)

    parser.add_argument(
        "--recon_use_noisy_z0", action="store_true", help="Use noisy z0 for recon instead of deterministic z0."
    )
    parser.add_argument("--decoder_prediction_type", type=str, default="x0", choices=["epsilon", "x0"])
    parser.add_argument("--save_recon_grid", type=str, default="stage1_recon_grid.png")
    parser.add_argument("--save_stage2_grid", type=str, default="stage2_samples_grid.png")
    parser.add_argument("--grid_items", type=int, default=8)

    parser.add_argument("--kid_subset_size", type=int, default=256)
    parser.add_argument("--kid_subsets", type=int, default=10)

    parser.add_argument("--pass_psnr", type=float, default=18.0)
    parser.add_argument("--pass_mae", type=float, default=0.10)
    parser.add_argument("--pass_kid", type=float, default=0.02)

    return parser.parse_args()


def _build_loader(data_dir: str, resolution: int, batch_size: int, num_workers: int):
    image_transform = transforms.Compose(
        [
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    dataset = ImageFolder(root=data_dir, transform=image_transform)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return dataset, loader


def _decode_image_diffusion(
    decoder: UNet2DModel,
    z0: torch.Tensor,
    *,
    resolution: int,
    num_sampling_steps: int,
    num_train_timesteps: int,
    schedule: str,
    lambda_min: float,
    lambda_max: float,
    decoder_prediction_type: str,
) -> torch.Tensor:
    bsz = z0.shape[0]
    z0_up = F.interpolate(z0, size=(resolution, resolution), mode="bilinear", align_corners=False)

    x_t = torch.randn(bsz, 3, resolution, resolution, device=z0.device, dtype=z0.dtype)
    t_grid = torch.linspace(1.0, 0.0, num_sampling_steps + 1, device=z0.device, dtype=z0.dtype)
    for i in range(num_sampling_steps):
        t_cur = t_grid[i].repeat(bsz)
        t_nxt = t_grid[i + 1].repeat(bsz)

        lambda_cur = ul_logsnr_schedule(t_cur, schedule_type=schedule, lambda_min=lambda_min, lambda_max=lambda_max)
        lambda_nxt = ul_logsnr_schedule(t_nxt, schedule_type=schedule, lambda_min=lambda_min, lambda_max=lambda_max)
        alpha_cur, sigma_cur = ul_alpha_sigma_from_logsnr(lambda_cur)
        alpha_nxt, sigma_nxt = ul_alpha_sigma_from_logsnr(lambda_nxt)

        timestep_idx = (t_cur * (num_train_timesteps - 1)).long().clamp(0, num_train_timesteps - 1)
        decoder_pred = decoder(torch.cat([x_t, z0_up], dim=1), timestep_idx).sample
        if decoder_prediction_type == "epsilon":
            eps_hat = decoder_pred
            x0_hat = (x_t - sigma_cur[:, None, None, None] * eps_hat) / alpha_cur[:, None, None, None].clamp_min(1e-5)
        else:
            x0_hat = decoder_pred
            eps_hat = (x_t - alpha_cur[:, None, None, None] * x0_hat) / sigma_cur[:, None, None, None].clamp_min(1e-5)
        x_t = alpha_nxt[:, None, None, None] * x0_hat + sigma_nxt[:, None, None, None] * eps_hat

    return x_t.clamp(-1, 1)


def _sample_stage2_latent(
    base_model: ULTwoStageBaseModel,
    *,
    batch_size: int,
    latent_channels: int,
    latent_size: int,
    num_sampling_steps: int,
    num_train_timesteps: int,
    schedule: str,
    lambda_min: float,
    lambda_max: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    z_t = torch.randn(batch_size, latent_channels, latent_size, latent_size, device=device, dtype=dtype)
    t_grid = torch.linspace(1.0, 0.0, num_sampling_steps + 1, device=device, dtype=dtype)
    for i in range(num_sampling_steps):
        t_cur = t_grid[i].repeat(batch_size)
        t_nxt = t_grid[i + 1].repeat(batch_size)

        lambda_cur = ul_logsnr_schedule(t_cur, schedule_type=schedule, lambda_min=lambda_min, lambda_max=lambda_max)
        lambda_nxt = ul_logsnr_schedule(t_nxt, schedule_type=schedule, lambda_min=lambda_min, lambda_max=lambda_max)
        alpha_cur, sigma_cur = ul_alpha_sigma_from_logsnr(lambda_cur)
        alpha_nxt, sigma_nxt = ul_alpha_sigma_from_logsnr(lambda_nxt)

        timestep_idx = (t_cur * (num_train_timesteps - 1)).long().clamp(0, num_train_timesteps - 1)
        dummy_labels = torch.zeros((batch_size,), device=device, dtype=torch.long)
        v_pred = base_model(z_t, timestep_idx, dummy_labels)

        z0_hat = alpha_cur[:, None, None, None] * z_t - sigma_cur[:, None, None, None] * v_pred
        eps_hat = sigma_cur[:, None, None, None] * z_t + alpha_cur[:, None, None, None] * v_pred
        z_t = alpha_nxt[:, None, None, None] * z0_hat + sigma_nxt[:, None, None, None] * eps_hat

    # Paper-ground-truth: hand off noisy latent at logsnr_0.
    return z_t


def _collect_inception_features(
    images_m11: torch.Tensor,
    inception: torch.nn.Module,
    *,
    device: torch.device,
) -> torch.Tensor:
    x = (images_m11 + 1.0) / 2.0
    x = x.clamp(0.0, 1.0)
    x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=x.dtype)[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=x.dtype)[None, :, None, None]
    x = (x - mean) / std
    with torch.no_grad():
        feats = inception(x)
    return feats.float().cpu()


def _kid_mmd2_unbiased(x: torch.Tensor, y: torch.Tensor) -> float:
    # Polynomial kernel used in KID.
    # x: [n, d], y: [m, d]
    n = x.shape[0]
    m = y.shape[0]
    d = x.shape[1]
    k_xx = ((x @ x.T) / d + 1.0) ** 3
    k_yy = ((y @ y.T) / d + 1.0) ** 3
    k_xy = ((x @ y.T) / d + 1.0) ** 3

    sum_xx = (k_xx.sum() - torch.diagonal(k_xx).sum()) / (n * (n - 1))
    sum_yy = (k_yy.sum() - torch.diagonal(k_yy).sum()) / (m * (m - 1))
    sum_xy = k_xy.mean()
    return (sum_xx + sum_yy - 2.0 * sum_xy).item()


def _compute_kid(
    real_feats: torch.Tensor,
    fake_feats: torch.Tensor,
    *,
    subset_size: int,
    subsets: int,
    seed: int,
) -> tuple[float, float]:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    n = min(real_feats.shape[0], fake_feats.shape[0])
    subset_size = min(subset_size, n)
    if subset_size < 2:
        raise ValueError("Need at least 2 samples to compute KID.")

    values = []
    for _ in range(subsets):
        idx_r = torch.randperm(real_feats.shape[0], generator=g)[:subset_size]
        idx_f = torch.randperm(fake_feats.shape[0], generator=g)[:subset_size]
        values.append(_kid_mmd2_unbiased(real_feats[idx_r], fake_feats[idx_f]))

    vals = torch.tensor(values, dtype=torch.float32)
    return vals.mean().item(), vals.std(unbiased=False).item()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    dataset, loader = _build_loader(args.train_data_dir, args.resolution, args.batch_size, args.num_workers)

    encoder = AutoencoderULEncoder.from_pretrained(args.stage1_encoder_path).to(device).eval()
    decoder = UNet2DModel.from_pretrained(args.stage1_decoder_path).to(device).eval()
    base_model = ULTwoStageBaseModel.from_pretrained(args.stage2_base_path).to(device).eval()

    latent_size = int(getattr(base_model.config, "latent_size", args.resolution // args.latent_downsample_factor))

    # Stage-1 reconstruction metrics.
    recon_count = 0
    recon_mae = []
    recon_mse = []
    vis_pairs = []

    for x, _ in loader:
        if recon_count >= args.num_recon_samples:
            break
        x = x.to(device)
        if recon_count + x.shape[0] > args.num_recon_samples:
            x = x[: args.num_recon_samples - recon_count]

        with torch.no_grad():
            z_clean = encoder.encode(x).latent
            lambda0 = torch.full((x.shape[0],), args.lambda_z_max, device=device, dtype=x.dtype)
            alpha0, sigma0 = ul_alpha_sigma_from_logsnr(lambda0)
            if args.recon_use_noisy_z0:
                eps0 = torch.randn_like(z_clean)
            else:
                eps0 = torch.zeros_like(z_clean)
            z0 = alpha0[:, None, None, None] * z_clean + sigma0[:, None, None, None] * eps0

            recon = _decode_image_diffusion(
                decoder,
                z0,
                resolution=args.resolution,
                num_sampling_steps=args.num_sampling_steps,
                num_train_timesteps=args.num_train_timesteps,
                schedule=args.decoder_schedule,
                lambda_min=args.lambda_x_min,
                lambda_max=args.lambda_x_max,
                decoder_prediction_type=args.decoder_prediction_type,
            )

        diff = recon - x
        mse = diff.square().mean(dim=(1, 2, 3))
        mae = diff.abs().mean(dim=(1, 2, 3))
        recon_mse.append(mse.cpu())
        recon_mae.append(mae.cpu())

        if len(vis_pairs) < args.grid_items:
            take = min(args.grid_items - len(vis_pairs), x.shape[0])
            for i in range(take):
                vis_pairs.append(torch.cat([x[i], recon[i]], dim=2).detach().cpu())

        recon_count += x.shape[0]

    recon_mse = torch.cat(recon_mse)
    recon_mae = torch.cat(recon_mae)
    recon_psnr = -10.0 * torch.log10(recon_mse.clamp_min(1e-12))

    if vis_pairs:
        recon_grid = make_grid([(p + 1.0) / 2.0 for p in vis_pairs], nrow=2)
        save_image(recon_grid, out_dir / args.save_recon_grid)

    # Stage-2 generation and realism metrics.
    gen_images = []
    gen_count = 0
    while gen_count < args.num_gen_samples:
        bsz = min(args.batch_size, args.num_gen_samples - gen_count)
        with torch.no_grad():
            z0_sampled = _sample_stage2_latent(
                base_model,
                batch_size=bsz,
                latent_channels=args.latent_channels,
                latent_size=latent_size,
                num_sampling_steps=args.num_sampling_steps,
                num_train_timesteps=args.num_train_timesteps,
                schedule=args.base_schedule,
                lambda_min=args.lambda_z_min,
                lambda_max=args.lambda_z_max,
                device=device,
                dtype=torch.float32,
            )
            x_gen = _decode_image_diffusion(
                decoder,
                z0_sampled,
                resolution=args.resolution,
                num_sampling_steps=args.num_sampling_steps,
                num_train_timesteps=args.num_train_timesteps,
                schedule=args.decoder_schedule,
                lambda_min=args.lambda_x_min,
                lambda_max=args.lambda_x_max,
                decoder_prediction_type=args.decoder_prediction_type,
            )
        gen_images.append(x_gen.cpu())
        gen_count += bsz

    gen_images = torch.cat(gen_images, dim=0)[: args.num_gen_samples]

    stage2_vis = make_grid(((gen_images[: args.grid_items] + 1.0) / 2.0), nrow=2)
    save_image(stage2_vis, out_dir / args.save_stage2_grid)

    # Real images for comparison features.
    real_images = []
    real_count = 0
    for x, _ in loader:
        if real_count >= args.num_gen_samples:
            break
        if real_count + x.shape[0] > args.num_gen_samples:
            x = x[: args.num_gen_samples - real_count]
        real_images.append(x)
        real_count += x.shape[0]
    real_images = torch.cat(real_images, dim=0)

    inception = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True, transform_input=False)
    inception.aux_logits = False
    inception.AuxLogits = None
    inception.fc = torch.nn.Identity()
    inception = inception.to(device).eval()

    real_feats = []
    fake_feats = []
    with torch.no_grad():
        for i in range(0, real_images.shape[0], args.batch_size):
            real_feats.append(
                _collect_inception_features(real_images[i : i + args.batch_size].to(device), inception, device=device)
            )
            fake_feats.append(
                _collect_inception_features(gen_images[i : i + args.batch_size].to(device), inception, device=device)
            )
    real_feats = torch.cat(real_feats, dim=0)
    fake_feats = torch.cat(fake_feats, dim=0)

    kid_mean, kid_std = _compute_kid(
        real_feats,
        fake_feats,
        subset_size=args.kid_subset_size,
        subsets=args.kid_subsets,
        seed=args.seed,
    )

    metrics = {
        "dataset_size": len(dataset),
        "num_recon_samples": int(recon_mse.numel()),
        "num_gen_samples": int(gen_images.shape[0]),
        "stage1": {
            "mae_mean": float(recon_mae.mean().item()),
            "mse_mean": float(recon_mse.mean().item()),
            "psnr_mean_db": float(recon_psnr.mean().item()),
            "psnr_median_db": float(recon_psnr.median().item()),
            "recon_uses_noisy_z0": bool(args.recon_use_noisy_z0),
        },
        "stage2": {
            "kid_mean": float(kid_mean),
            "kid_std": float(kid_std),
        },
        "passes": {
            "stage1_psnr": bool(recon_psnr.mean().item() >= args.pass_psnr),
            "stage1_mae": bool(recon_mae.mean().item() <= args.pass_mae),
            "stage2_kid": bool(kid_mean <= args.pass_kid),
        },
        "thresholds": {
            "pass_psnr": float(args.pass_psnr),
            "pass_mae": float(args.pass_mae),
            "pass_kid": float(args.pass_kid),
        },
        "artifacts": {
            "recon_grid": str((out_dir / args.save_recon_grid).resolve()),
            "stage2_grid": str((out_dir / args.save_stage2_grid).resolve()),
        },
    }
    metrics["passes"]["all"] = all(metrics["passes"].values())

    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
