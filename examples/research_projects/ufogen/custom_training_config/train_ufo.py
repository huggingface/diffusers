# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT.
"""
import torch

# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
from diffusers.models import AutoencoderKL
from accelerate import Accelerator

from diffusion import create_diffusion
from torchvision.utils import save_image
import torch.nn.functional as F

from tensorboardX import SummaryWriter


#################################################################################
#                             Training Helper Functions                         #
#################################################################################


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def optim_warmup(warmup_iters, lr, step, optim):
    lr = lr * float(step) / warmup_iters
    for param_group in optim.param_groups:
        param_group['lr'] = lr


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


class CustomDataset(Dataset):
    def __init__(self, features_dir, labels_dir):
        self.features_dir = features_dir
        self.labels_dir = labels_dir

        self.features_files = sorted(os.listdir(features_dir))
        self.labels_files = sorted(os.listdir(labels_dir))

    def __len__(self):
        assert len(self.features_files) == len(self.labels_files), \
            "Number of feature files and label files should be same"
        return len(self.features_files)

    def __getitem__(self, idx):
        feature_file = self.features_files[idx]
        label_file = self.labels_files[idx]

        features = np.load(os.path.join(self.features_dir, feature_file))
        labels = np.load(os.path.join(self.labels_dir, label_file))
        return torch.from_numpy(features), torch.from_numpy(labels)


#################################################################################
#                                  Training Loop                                #
#################################################################################

def sample_from_model(model, args, diffusion, num_times, x_init, y):
    x = x_init
    with torch.no_grad():
        for i in reversed(range(num_times)):
            t = ((torch.full((x.size(0),), i + 1, dtype=torch.float).to(x.device) / num_times)*(diffusion.num_timesteps-1)).long()
            t_minus = (t - (diffusion.num_timesteps-1)/num_times).long()
            model_out = model(x, t, y)
            pred_xstart = diffusion.pred_xstart(model_out, x, t, clip_denoised=False, denoised_fn=None, model_kwargs=None)["pred_xstart"]
            eps = torch.randn_like(x)
            x = diffusion.q_sample(pred_xstart, t_minus, eps)
    return pred_xstart.detach()

def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    from models import DiT_models

    # Setup accelerator:
    accelerator = Accelerator()
    device = accelerator.device

    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    # Setup an experiment folder:
    if accelerator.is_main_process:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")

        log_writer = SummaryWriter(logdir=os.path.join(experiment_dir, 'logs'))

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
    )
    model_dis = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
    )
    # Note that parameter initialization is done within the DiT constructor
    model = model.to(device)
    model_dis = model_dis.to(device)
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
    if accelerator.is_main_process:
        logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=0.0, eps=1e-12)
    opt_dis = torch.optim.Adam(model_dis.parameters(), lr=1e-4, betas=(0.0, 0.999), weight_decay=0.0, eps=1e-12)

    # Setup data:
    features_dir = f"{args.feature_path}/imagenet256_features"
    labels_dir = f"{args.feature_path}/imagenet256_labels"
    dataset = CustomDataset(features_dir, labels_dir)
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // accelerator.num_processes),
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    if accelerator.is_main_process:
        logger.info(f"Dataset contains {len(dataset):,} images ({args.feature_path})")
    # Prepare models for training:
    if os.path.exists(args.pretrained_ckpt_dir):
        ckpt = torch.load(args.pretrained_ckpt_dir)
        missing_keys, unexpected_keys = model.load_state_dict(ckpt, strict=False)
        if accelerator.is_main_process:
            print("missing keys: ", missing_keys)
            print("unexpected keys: ", unexpected_keys)
        model_dis.load_state_dict(ckpt, strict=False)
        model_dis.zero_init_last_layer()
        print("load pretrained ckpt")
        del ckpt
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    model_dis.train()
    ema.eval()  # EMA model should always be in eval mode
    requires_grad(ema, False)
    model, model_dis, opt, opt_dis, loader = accelerator.prepare(model, model_dis, opt, opt_dis, loader)

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    if accelerator.is_main_process:
        logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        if accelerator.is_main_process:
            logger.info(f"Beginning epoch {epoch}...")
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            x = x.squeeze(dim=1)
            y = y.squeeze(dim=1)
            model_kwargs = dict(y=y)

            if train_steps % 500 == 0 and accelerator.is_main_process:
                with torch.no_grad():
                    noise = torch.randn_like(x[:16, ...])
                    samples = sample_from_model(ema, args, diffusion, 1, noise, y[:16, ...])
                    samples = vae.decode(samples / 0.18215).sample
                    save_image(samples, os.path.join(experiment_dir, str(train_steps)+"_1sample.png"),
                    nrow=4, normalize=True, value_range=(-1, 1))
                    samples = sample_from_model(ema, args, diffusion, 2, noise, y[:16, ...])
                    samples = vae.decode(samples / 0.18215).sample
                    save_image(samples, os.path.join(experiment_dir, str(train_steps)+"_2sample.png"),
                    nrow=4, normalize=True, value_range=(-1, 1))
                    samples = sample_from_model(ema, args, diffusion, 4, noise, y[:16, ...])
                    samples = vae.decode(samples / 0.18215).sample
                    save_image(samples, os.path.join(experiment_dir, str(train_steps)+"_4sample.png"),
                    nrow=4, normalize=True, value_range=(-1, 1))
                    reals = vae.decode(x[:16, ...] / 0.18215).sample
                    save_image(reals, os.path.join(experiment_dir, str(train_steps)+"_real.png"),
                    nrow=4, normalize=True, value_range=(-1, 1))

            ########## D step
            if train_steps % 2 == 0:
                requires_grad(model_dis, True)

                t = torch.randint(1, diffusion.num_timesteps, (x.shape[0],), device=device)
                t_minus = (t - diffusion.num_timesteps/args.num_times).clamp(min=0.0).long()

                noise = torch.randn_like(x)
                x_t = diffusion.q_sample(x, t, noise)

                x_t_minus = diffusion.q_sample(x, t_minus, torch.randn_like(x))
                
                model_out = model(x_t, t, **model_kwargs)
                pred_xstart = diffusion.pred_xstart(model_out, x_t, t, clip_denoised=False, denoised_fn=None, model_kwargs=None)["pred_xstart"]
                eps = torch.randn_like(pred_xstart)
                fake_x_t_minus = diffusion.q_sample(pred_xstart, t_minus, eps)

                dis_input = torch.cat([fake_x_t_minus.detach(), x_t_minus], dim=0)
                t_input = torch.cat([t_minus]*2, dim=0)
                model_kwargs_input = {}
                for k in model_kwargs.keys():
                    model_kwargs_input[k] = torch.cat([model_kwargs[k]]*2, dim=0)
                dis_out = model_dis(dis_input, t_input, **model_kwargs_input)
                fake_dis, real_dis = dis_out.chunk(2, dim=0)
                loss_dis_fake = F.binary_cross_entropy_with_logits(fake_dis.clamp(min=-0.5), torch.zeros_like(fake_dis))
                loss_dis_real = F.binary_cross_entropy_with_logits(real_dis.clamp(max=0.5), torch.ones_like(real_dis))
                loss_dis = loss_dis_fake + loss_dis_real

                opt_dis.zero_grad()
                accelerator.backward(loss_dis)
                grad_norm_d = accelerator.clip_grad_norm_(model_dis.parameters(), 1.0)
                opt_dis.step()

            ########## G step
            requires_grad(model_dis, False)
            
            t = torch.randint(1, diffusion.num_timesteps, (x.shape[0],), device=device)
            t_minus = (t - diffusion.num_timesteps/args.num_times).clamp(min=0.0).long()
            noise = torch.randn_like(x)
            x_t = diffusion.q_sample(x, t, noise)
            loss_terms = diffusion.training_losses(model, x, t, model_kwargs)
            loss_c = loss_terms["loss"].mean()
            pred_xstart = loss_terms["pred_xstart"]
            eps = torch.randn_like(x)
            fake_x_t_minus  = diffusion.q_sample(pred_xstart, t_minus, eps)

            fake_dis = model_dis(fake_x_t_minus, t_minus, **model_kwargs)
            loss_gen = F.binary_cross_entropy_with_logits(fake_dis.clamp(max=0.5), torch.ones_like(fake_dis))

            opt.zero_grad()
            accelerator.backward(loss_gen + loss_c)
            grad_norm_g = accelerator.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            
            update_ema(ema, model, 0.99)

            # Log loss values:
            running_loss += loss_c.item()
            log_steps += 1
            train_steps += 1
            if train_steps % 2 == 0:
                if accelerator.is_main_process:
                    log_writer.add_scalar('G_Loss', loss_gen.item(), global_step=train_steps)
                    log_writer.add_scalar('D_Loss', loss_dis.item(), global_step=train_steps)
                    log_writer.add_scalar('grad_g', grad_norm_g.item(), global_step=train_steps)
                    log_writer.add_scalar('grad_d', grad_norm_d.item(), global_step=train_steps)
                    log_writer.add_scalar('condition', loss_c.item(), global_step=train_steps)
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                avg_loss = avg_loss.item() / accelerator.num_processes
                if accelerator.is_main_process:
                    logger.info(
                        f"(step={train_steps:07d}) loss_c: {loss_c.item():.4f}, loss_d: {loss_dis.item():.4f}, loss_g: {loss_gen.item():.4f}, grad_norm_g: {grad_norm_g.item():.4f}, grad_norm_d: {grad_norm_d.item():.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save DiT checkpoint:

            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if accelerator.is_main_process:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    if accelerator.is_main_process:
        logger.info("Done!")


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-path", type=str, default="features")
    parser.add_argument("--results-dir", type=str, default="results_ufo")
    parser.add_argument("--pretrained-ckpt-dir", type=str, default="")
    parser.add_argument("--model", type=str, default="DiT_XL_2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--num_times", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--ckpt-every", type=int, default=5_000)
    args = parser.parse_args()
    main(args)
