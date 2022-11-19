import copy
from random import sample
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
from tqdm.auto import tqdm
from accelerate import Accelerator
from diffusers import DiffusionPipeline
from diffusers.optimization import get_scheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.pipelines.ddpm import DDPMPipeline
from diffusers.pipelines.ddim import DDIMPipeline
from diffusers.training_utils import EMAModel


def logsnr_schedule(t, logsnr_min=-20, logsnr_max=20):
    logsnr_min = torch.tensor(logsnr_min, dtype=torch.float32)
    logsnr_max = torch.tensor(logsnr_max, dtype=torch.float32)
    b = torch.arctan(torch.exp(-0.5 * logsnr_max))
    a = torch.arctan(torch.exp(-0.5 * logsnr_min)) - b
    return -2.0 * torch.log(torch.tan(a * t + b))


def continuous_to_discrete_time(u, num_timesteps):
    return (u * (num_timesteps - 1)).float().round().long()


def predict_x_from_v(*, z, v, logsnr):
    logsnr = utils.broadcast_from_left(logsnr, z.shape)
    alpha_t = torch.sqrt(F.sigmoid(logsnr))
    sigma_t = torch.sqrt(F.sigmoid(-logsnr))
    return alpha_t * z - sigma_t * v


def alpha_sigma_from_logsnr(logsnr):
    alpha_t = torch.sqrt(F.sigmoid(logsnr))
    sigma_t = torch.sqrt(F.sigmoid(-logsnr))
    return alpha_t, sigma_t


class DistillationPipeline(DiffusionPipeline):
    def __init__(self):
        pass

    def __call__(
        self,
        teacher,
        n_teacher_trainsteps,
        train_data,
        epochs=100,
        lr=3e-4,
        batch_size=64,
        gamma=0,
        gradient_accumulation_steps=1,
        mixed_precision="no",
        adam_beta1=0.95,
        adam_beta2=0.999,
        adam_weight_decay=0.001,
        adam_epsilon=1e-08,
        ema_inv_gamma=0.9999,
        ema_power=3 / 4,
        ema_max_decay=0.9999,
        use_ema=True,
        permute_samples=(0, 1, 2, 3),
        generator=None,
        accelerator=None,
        sample_every: int = None,
        sample_path: str = "distillation_samples",
    ):
        # Initialize our accelerator for training
        os.makedirs(os.path.join(sample_path, f"{n_teacher_trainsteps}"), exist_ok=True)
        if accelerator is None:
            accelerator = Accelerator(
                gradient_accumulation_steps=gradient_accumulation_steps,
                mixed_precision=mixed_precision,
            )

        if accelerator.is_main_process:
            run = "distill"
            accelerator.init_trackers(run)

        # Setup a dataloader with the provided train data
        train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

        # Setup the noise schedulers for the teacher and student
        teacher_scheduler = DDIMScheduler(
            num_train_timesteps=n_teacher_trainsteps,
            beta_schedule="squaredcos_cap_v2",
            variance_type="v_diffusion",
            prediction_type="v",
        )
        student_scheduler = DDIMScheduler(
            num_train_timesteps=n_teacher_trainsteps // 2,
            beta_schedule="squaredcos_cap_v2",
            variance_type="v_diffusion",
            prediction_type="v",
        )

        # Initialize the student model as a direct copy of the teacher
        student = copy.deepcopy(teacher)
        student.load_state_dict(teacher.state_dict())
        student = accelerator.prepare(student)
        student.train()
        teacher.eval()

        # Setup the optimizer for the student
        optimizer = torch.optim.AdamW(
            student.parameters(),
            lr=lr,
            # betas=(adam_beta1, adam_beta2),
            # weight_decay=adam_weight_decay,
            # eps=adam_epsilon,
        )
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=np.ceil((epochs * len(train_dataloader)) // gradient_accumulation_steps),
        )

        # Let accelerate handle moving the model to the correct device
        (
            teacher,
            student,
            optimizer,
            lr_scheduler,
            train_data,
            teacher_scheduler,
            student_scheduler,
        ) = accelerator.prepare(
            teacher, student, optimizer, lr_scheduler, train_data, teacher_scheduler, student_scheduler
        )
        if not generator:
            generator = torch.Generator().manual_seed(0)

        # generator = accelerator.prepare(generator)
        ema_model = EMAModel(
            student,
            inv_gamma=ema_inv_gamma,
            power=ema_power,
            max_value=ema_max_decay,
        )
        global_step = 0

        # run pipeline in inference (sample random noise and denoise) on our teacher model as a baseline
        pipeline = DDIMPipeline(
            unet=teacher,
            scheduler=teacher_scheduler,
        )

        images = pipeline(batch_size=4, generator=torch.manual_seed(0)).images

        # denormalize the images and save to tensorboard
        # images_processed = (images * 255).round().astype("uint8")
        for sample_number, img in enumerate(images):

            img.save(os.path.join(sample_path, f"{n_teacher_trainsteps}", f"baseline_sample_{sample_number}.png"))

        # Train the student
        for epoch in range(epochs):
            progress_bar = tqdm(total=len(train_data) // batch_size, disable=not accelerator.is_local_main_process)
            progress_bar.set_description(f"Epoch {epoch}")
            for batch in train_dataloader:
                with accelerator.accumulate(student):
                    if isinstance(batch, dict):
                        batch = batch["images"]
                    batch = batch.to(accelerator.device)
                    noise = torch.randn(batch.shape, generator=generator).to(accelerator.device)
                    bsz = batch.shape[0]
                    # Sample a random timestep for each image

                    u = torch.rand(size=(bsz,), generator=generator).to(accelerator.device)
                    u_1 = u - (0.5 / (n_teacher_trainsteps // 2))
                    u_2 = u - (1 / (n_teacher_trainsteps // 2))
                    # logsnr = logsnr_schedule(u)
                    # alpha_t, sigma_t = alpha_sigma_from_logsnr(logsnr)
                    with torch.no_grad():
                        # Add noise to the image based on noise scheduler a t=timesteps
                        timesteps = continuous_to_discrete_time(u, n_teacher_trainsteps)
                        alpha_t, sigma_t = teacher_scheduler.get_alpha_sigma(batch, timesteps, accelerator.device)
                        z_t = alpha_t * batch + sigma_t * noise
                        # z_t = batch * torch.sqrt(F.sigmoid(logsnr)) + noise * torch.sqrt(F.sigmoid(-logsnr))

                        # teach_out_start = teacher(z_t, continuous_to_discrete_time(u, n_teacher_trainsteps))
                        # x_pred = predict_x_from_v(teach_out_start)
                        # Take the first diffusion step with the teacher
                        v_pred_t = teacher(z_t.permute(*permute_samples), timesteps).sample.permute(*permute_samples)

                        # reconstruct the image at timesteps using v diffusion
                        x_teacher_z_t = alpha_t * z_t - sigma_t * v_pred_t
                        # eps = (z - alpha*x)/sigma.
                        eps_pred = (z_t - alpha_t * x_teacher_z_t) / sigma_t

                        # Add noise to the image based on noise scheduler a t=timesteps-1, to prepare for the next diffusion step
                        timesteps = continuous_to_discrete_time(u_1, n_teacher_trainsteps)
                        alpha_t_prime, sigma_t_prime = teacher_scheduler.get_alpha_sigma(
                            batch, timesteps, accelerator.device
                        )
                        z_mid = alpha_t_prime * x_teacher_z_t + sigma_t_prime * eps_pred
                        # Take the second diffusion step with the teacher
                        v_pred_mid = teacher(z_mid.permute(*permute_samples), timesteps).sample.permute(
                            *permute_samples
                        )
                        x_pred_mid = alpha_t_prime * z_mid - sigma_t_prime * v_pred_mid

                        eps_pred = (z_mid - alpha_t_prime * x_pred_mid) / sigma_t_prime

                        timesteps = continuous_to_discrete_time(u_2, n_teacher_trainsteps)
                        alpha_t_prime2, sigma_t_prime2 = teacher_scheduler.get_alpha_sigma(
                            batch, timesteps, accelerator.device
                        )
                        z_teacher = alpha_t_prime2 * x_pred_mid + sigma_t_prime2 * eps_pred
                        sigma_frac = sigma_t / sigma_t_prime2

                        x_target = (z_teacher - sigma_frac * z_t) / (alpha_t_prime2 - sigma_frac * alpha_t)
                        eps_target = (z_teacher - alpha_t_prime2 * x_target) / sigma_t_prime2
                        v_target = alpha_t * eps_target - sigma_t * x_target

                    timesteps = continuous_to_discrete_time(u_2, n_teacher_trainsteps // 2)
                    noise_pred = student(z_t.permute(*permute_samples), timesteps).sample.permute(*permute_samples)
                    w = torch.pow(1 + alpha_t_prime2 / sigma_t_prime2, gamma)
                    loss = F.mse_loss(noise_pred * w, v_target * w)
                    accelerator.backward(loss)

                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(student.parameters(), 1.0)
                    optimizer.step()
                    lr_scheduler.step()
                    if use_ema:
                        ema_model.step(student)
                    optimizer.zero_grad()

                    # Checks if the accelerator has performed an optimization step behind the scenes
                    if accelerator.sync_gradients:
                        progress_bar.update(1)
                        global_step += 1

                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
                if use_ema:
                    logs["ema_decay"] = ema_model.decay
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)
            progress_bar.close()
            if sample_every is not None:
                if (epoch + 1) % sample_every == 0:
                    new_scheduler = DDIMScheduler(
                        num_train_timesteps=n_teacher_trainsteps // 2,
                        beta_schedule="squaredcos_cap_v2",
                        variance_type="v_diffusion",
                        prediction_type="v",
                    )
                    pipeline = DDIMPipeline(
                        unet=accelerator.unwrap_model(ema_model.averaged_model if use_ema else student),
                        scheduler=new_scheduler,
                    )

                    # run pipeline in inference (sample random noise and denoise)
                    images = pipeline(
                        batch_size=4,
                        generator=torch.manual_seed(0),
                        num_inference_steps=n_teacher_trainsteps // 2,
                    ).images

                    # denormalize the images and save to tensorboard
                    for sample_number, img in enumerate(images):
                        img.save(
                            os.path.join(
                                sample_path, f"{n_teacher_trainsteps}", f"epoch_{epoch}_sample_{sample_number}.png"
                            )
                        )
            accelerator.wait_for_everyone()
        return student, ema_model, accelerator
