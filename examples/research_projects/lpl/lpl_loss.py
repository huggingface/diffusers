# Copyright 2025 Berrada et al.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_tensor(in_feat, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(in_feat**2, dim=1, keepdim=True))
    return in_feat / (norm_factor + eps)


def cross_normalize(input, target, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(target**2, dim=1, keepdim=True))
    return input / (norm_factor + eps), target / (norm_factor + eps)


def remove_outliers(feat, down_f=1, opening=5, closing=3, m=100, quant=0.02):
    opening = int(np.ceil(opening / down_f))
    closing = int(np.ceil(closing / down_f))
    if opening == 2:
        opening = 3
    if closing == 2:
        closing = 1

    # replace quantile with kth value here.
    feat_flat = feat.flatten(-2, -1)
    k1, k2 = int(feat_flat.shape[-1] * quant), int(feat_flat.shape[-1] * (1 - quant))
    q1 = feat_flat.kthvalue(k1, dim=-1).values[..., None, None]
    q2 = feat_flat.kthvalue(k2, dim=-1).values[..., None, None]

    m = 2 * feat_flat.std(-1)[..., None, None].detach()
    mask = (q1 - m < feat) * (feat < q2 + m)

    # dilate the mask.
    mask = nn.MaxPool2d(kernel_size=closing, stride=1, padding=(closing - 1) // 2)(mask.float())  # closing
    mask = (-nn.MaxPool2d(kernel_size=opening, stride=1, padding=(opening - 1) // 2)(-mask)).bool()  # opening
    feat = feat * mask
    return mask, feat


class LatentPerceptualLoss(nn.Module):
    def __init__(
        self,
        vae,
        loss_type="mse",
        grad_ckpt=True,
        pow_law=False,
        norm_type="default",
        num_mid_blocks=4,
        feature_type="feature",
        remove_outliers=True,
    ):
        super().__init__()
        self.vae = vae
        self.decoder = self.vae.decoder
        # Store scaling factors as tensors on the correct device
        device = next(self.vae.parameters()).device

        # Get scaling factors with proper defaults and handle None values
        scale_factor = getattr(self.vae.config, "scaling_factor", None)
        shift_factor = getattr(self.vae.config, "shift_factor", None)

        # Convert to tensors with proper defaults
        self.scale = torch.tensor(1.0 if scale_factor is None else scale_factor, device=device)
        self.shift = torch.tensor(0.0 if shift_factor is None else shift_factor, device=device)

        self.gradient_checkpointing = grad_ckpt
        self.pow_law = pow_law
        self.norm_type = norm_type.lower()
        self.outlier_mask = remove_outliers
        self.last_feature_stats = []  # Store feature statistics for logging

        assert feature_type in ["feature", "image"]
        self.feature_type = feature_type

        assert self.norm_type in ["default", "shared", "batch"]
        assert num_mid_blocks >= 0 and num_mid_blocks <= 4
        self.n_blocks = num_mid_blocks

        assert loss_type in ["mse", "l1"]
        if loss_type == "mse":
            self.loss_fn = nn.MSELoss(reduction="none")
        elif loss_type == "l1":
            self.loss_fn = nn.L1Loss(reduction="none")

    def get_features(self, z, latent_embeds=None, disable_grads=False):
        with torch.set_grad_enabled(not disable_grads):
            if self.gradient_checkpointing and not disable_grads:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                features = []
                upscale_dtype = next(iter(self.decoder.up_blocks.parameters())).dtype
                sample = z
                sample = self.decoder.conv_in(sample)

                # middle
                sample = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.decoder.mid_block),
                    sample,
                    latent_embeds,
                    use_reentrant=False,
                )
                sample = sample.to(upscale_dtype)
                features.append(sample)

                # up
                for up_block in self.decoder.up_blocks[: self.n_blocks]:
                    sample = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(up_block),
                        sample,
                        latent_embeds,
                        use_reentrant=False,
                    )
                    features.append(sample)
                return features
            else:
                features = []
                upscale_dtype = next(iter(self.decoder.up_blocks.parameters())).dtype
                sample = z
                sample = self.decoder.conv_in(sample)

                # middle
                sample = self.decoder.mid_block(sample, latent_embeds)
                sample = sample.to(upscale_dtype)
                features.append(sample)

                # up
                for up_block in self.decoder.up_blocks[: self.n_blocks]:
                    sample = up_block(sample, latent_embeds)
                    features.append(sample)
                return features

    def get_loss(self, input, target, get_hist=False):
        if self.feature_type == "feature":
            inp_f = self.get_features(self.shift + input / self.scale)
            tar_f = self.get_features(self.shift + target / self.scale, disable_grads=True)
            losses = []
            self.last_feature_stats = []  # Reset feature stats

            for i, (x, y) in enumerate(zip(inp_f, tar_f, strict=False)):
                my = torch.ones_like(y).bool()
                outlier_ratio = 0.0

                if self.outlier_mask:
                    with torch.no_grad():
                        if i == 2:
                            my, y = remove_outliers(y, down_f=2)
                            outlier_ratio = 1.0 - my.float().mean().item()
                        elif i in [3, 4, 5]:
                            my, y = remove_outliers(y, down_f=1)
                            outlier_ratio = 1.0 - my.float().mean().item()

                # Store feature statistics before normalization
                with torch.no_grad():
                    stats = {
                        "mean": y.mean().item(),
                        "std": y.std().item(),
                        "outlier_ratio": outlier_ratio,
                    }
                    self.last_feature_stats.append(stats)

                # normalize feature tensors
                if self.norm_type == "default":
                    x = normalize_tensor(x)
                    y = normalize_tensor(y)
                elif self.norm_type == "shared":
                    x, y = cross_normalize(x, y, eps=1e-6)

                term_loss = self.loss_fn(x, y) * my
                # reduce loss term
                loss_f = 2 ** (-min(i, 3)) if self.pow_law else 1.0
                term_loss = term_loss.sum((2, 3)) * loss_f / my.sum((2, 3))
                losses.append(term_loss.mean((1,)))

            if get_hist:
                return losses
            else:
                loss = sum(losses)
                return loss / len(inp_f)
        elif self.feature_type == "image":
            inp_f = self.vae.decode(input / self.scale).sample
            tar_f = self.vae.decode(target / self.scale).sample
            return F.mse_loss(inp_f, tar_f)

    def get_first_conv(self, z):
        sample = self.decoder.conv_in(z)
        return sample

    def get_first_block(self, z):
        sample = self.decoder.conv_in(z)
        sample = self.decoder.mid_block(sample)
        for resnet in self.decoder.up_blocks[0].resnets:
            sample = resnet(sample, None)
        return sample

    def get_first_layer(self, input, target, target_layer="conv"):
        if target_layer == "conv":
            feat_in = self.get_first_conv(input)
            with torch.no_grad():
                feat_tar = self.get_first_conv(target)
        else:
            feat_in = self.get_first_block(input)
            with torch.no_grad():
                feat_tar = self.get_first_block(target)

        feat_in, feat_tar = cross_normalize(feat_in, feat_tar)

        return F.mse_loss(feat_in, feat_tar, reduction="mean")
