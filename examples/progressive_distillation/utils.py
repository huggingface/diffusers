

from dataclasses import dataclass
from torchvision.transforms import (
    CenterCrop,
    Compose,
    InterpolationMode,
    Normalize,
    RandomHorizontalFlip,
    Resize,
    ToTensor,
)
import torch.nn.functional as F

import torch
from diffusers import UNet2DModel, DDIMScheduler, DDPMScheduler
from accelerate import Accelerator
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from tqdm import tqdm

@dataclass
class DiffusionTrainingArgs:
    resolution: int = 64
    mixed_precision: str = "fp16"
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    lr_scheduler: str = "cosine"
    lr_warmup_steps: int = 500
    adam_beta1: float = 0.95
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1e-6
    adam_epsilon: float = 1e-08
    use_ema: bool = True
    ema_inv_gamma: float = 1.0
    ema_power: float = 3 / 4
    ema_max_decay: float = 0.9999
    batch_size: int = 64
    num_epochs: int = 500

def get_train_transforms(training_config):
    return Compose(
    [
        Resize(training_config.resolution, interpolation=InterpolationMode.BILINEAR),
        CenterCrop(training_config.resolution),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize([0.5], [0.5]),
    ]
)

def get_unet(training_config):
    return UNet2DModel(
        sample_size=training_config.resolution,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(128, 128, 256, 256, 512, 512),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )


def distill(teacher, n, train_image, training_config, epochs=100, lr=3e-4, batch_size=16, gamma=0, generator=None):
    if generator is None:
        generator = torch.manual_seed(0)
    accelerator = Accelerator(
    gradient_accumulation_steps=training_config.gradient_accumulation_steps,
    mixed_precision=training_config.mixed_precision,
)
    if accelerator.is_main_process:
        run = "distill"
        accelerator.init_trackers(run)
    teacher_scheduler = DDPMScheduler(num_train_timesteps=n, beta_schedule="squaredcos_cap_v2")
    student_scheduler = DDPMScheduler(num_train_timesteps=n // 2, beta_schedule="squaredcos_cap_v2")
    student = get_unet(training_config)
    student.load_state_dict(teacher.state_dict())
    student = accelerator.prepare(student)
    student.train()
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=lr,
        betas=(training_config.adam_beta1, training_config.adam_beta2),
        weight_decay=0.001,
        eps=training_config.adam_epsilon,
    )
    lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=(epochs) // training_config.gradient_accumulation_steps,
)
    teacher, student, optimizer, lr_scheduler, train_image, teacher_scheduler, student_scheduler = accelerator.prepare(
    teacher, student, optimizer, lr_scheduler, train_image,teacher_scheduler, student_scheduler
)
    ema_model = EMAModel(student, inv_gamma=training_config.ema_inv_gamma, power=training_config.ema_power, max_value=training_config.ema_max_decay)
    global_step = 0
    for epoch in range(epochs):
        progress_bar = tqdm(total=1, disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        batch = train_image.unsqueeze(0).repeat(
            batch_size, 1, 1, 1
        ).to(accelerator.device)
        with accelerator.accumulate(student):
            noise = torch.randn(batch.shape).to(accelerator.device)
            bsz = batch.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, student_scheduler.config.num_train_timesteps, (bsz,), device=batch.device
            ).long() * 2
            with torch.no_grad():
                # Add noise to the image based on noise scheduler a t=timesteps
                alpha_t, sigma_t = teacher_scheduler.get_alpha_sigma(batch, timesteps + 1, accelerator.device)
                z_t = alpha_t * batch + sigma_t * noise

                # Take the first diffusion step with the teacher
                noise_pred_t = teacher(z_t, timesteps + 1).sample
                x_teacher_z_t = (alpha_t * z_t - sigma_t * noise_pred_t).clip(-1, 1)

                # Add noise to the image based on noise scheduler a t=timesteps-1, to prepare for the next diffusion step
                alpha_t_prime, sigma_t_prime = teacher_scheduler.get_alpha_sigma(batch, timesteps, accelerator.device)
                z_t_prime = alpha_t_prime * x_teacher_z_t + (sigma_t_prime / sigma_t) * (z_t - alpha_t * x_teacher_z_t)
                # Take the second diffusion step with the teacher
                noise_pred_t_prime = teacher(z_t_prime.float(), timesteps).sample
                rec_t_prime = (alpha_t_prime * z_t_prime - sigma_t_prime * noise_pred_t_prime).clip(-1, 1)

                # V prediction per Appendix D
                alpha_t_prime2, sigma_t_prime2 = student_scheduler.get_alpha_sigma(batch, timesteps // 2, accelerator.device)
                x_teacher_z_t_prime = (z_t - alpha_t_prime2 * rec_t_prime) / sigma_t_prime2
                z_t_prime_2 = alpha_t_prime2 * x_teacher_z_t_prime - sigma_t_prime2 * rec_t_prime

            noise_pred = student(z_t, timesteps).sample
            w = torch.pow(1 + alpha_t_prime2 / sigma_t_prime2, gamma)
            loss = F.mse_loss(noise_pred * w, z_t_prime_2 * w)
            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            if training_config.use_ema:
                ema_model.step(student)
            optimizer.zero_grad()

        # Checks if the accelerator has performed an optimization step behind the scenes
        if accelerator.sync_gradients:
            progress_bar.update(1)
            global_step += 1

        logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
        if training_config.use_ema:
            logs["ema_decay"] = ema_model.decay
        progress_bar.set_postfix(**logs)
        accelerator.log(logs, step=global_step)
        progress_bar.close()

        accelerator.wait_for_everyone()
    return student, ema_model, accelerator
