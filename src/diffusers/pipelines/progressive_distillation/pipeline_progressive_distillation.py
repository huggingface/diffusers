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
from diffusers.pipelines.ddpm import DDPMPipeline
from diffusers.training_utils import EMAModel


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
        teacher_scheduler = DDPMScheduler(
            num_train_timesteps=n_teacher_trainsteps,
            beta_schedule="squaredcos_cap_v2",
            variance_type="fixed_small_log",
        )
        student_scheduler = DDPMScheduler(
            num_train_timesteps=n_teacher_trainsteps // 2,
            beta_schedule="squaredcos_cap_v2",
            variance_type="fixed_small_log",
        )

        # Initialize the student model as a direct copy of the teacher
        student = copy.deepcopy(teacher)
        student.load_state_dict(teacher.state_dict())
        student = accelerator.prepare(student)
        student.train()

        # Setup the optimizer for the student
        optimizer = torch.optim.AdamW(
            student.parameters(),
            lr=lr,
            betas=(adam_beta1, adam_beta2),
            weight_decay=adam_weight_decay,
            eps=adam_epsilon,
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
        if generator:
            generator = accelerator.prepare(generator)
        ema_model = EMAModel(
            student,
            inv_gamma=ema_inv_gamma,
            power=ema_power,
            max_value=ema_max_decay,
        )
        global_step = 0

        # run pipeline in inference (sample random noise and denoise) on our teacher model as a baseline
        pipeline = DDPMPipeline(
            unet=teacher,
            scheduler=teacher_scheduler,
        )

        images = pipeline(batch_size=4, output_type="numpy", generator=torch.manual_seed(0)).images

        # denormalize the images and save to tensorboard
        images_processed = (images * 255).round().astype("uint8")
        for sample_number, img in enumerate(images_processed):
            img = Image.fromarray(img)

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
                    timesteps = (
                        torch.randint(
                            0,
                            student_scheduler.config.num_train_timesteps,
                            (bsz,),
                            device=batch.device,
                            generator=generator,
                        ).long()
                        * 2
                    )
                    with torch.no_grad():
                        # Add noise to the image based on noise scheduler a t=timesteps
                        alpha_t, sigma_t = teacher_scheduler.get_alpha_sigma(batch, timesteps + 1, accelerator.device)
                        z_t = alpha_t * batch + sigma_t * noise

                        # Take the first diffusion step with the teacher
                        noise_pred_t = teacher(z_t.permute(*permute_samples), timesteps + 1).sample.permute(
                            *permute_samples
                        )
                        x_teacher_z_t = (alpha_t * z_t - sigma_t * noise_pred_t).clip(-1, 1)

                        # Add noise to the image based on noise scheduler a t=timesteps-1, to prepare for the next diffusion step
                        alpha_t_prime, sigma_t_prime = teacher_scheduler.get_alpha_sigma(
                            batch, timesteps, accelerator.device
                        )
                        z_t_prime = alpha_t_prime * x_teacher_z_t + (sigma_t_prime / sigma_t) * (
                            z_t - alpha_t * x_teacher_z_t
                        )
                        # Take the second diffusion step with the teacher
                        noise_pred_t_prime = teacher(z_t_prime.permute(*permute_samples), timesteps).sample.permute(
                            *permute_samples
                        )
                        rec_t_prime = (alpha_t_prime * z_t_prime - sigma_t_prime * noise_pred_t_prime).clip(-1, 1)

                        # V prediction per Appendix D
                        alpha_t_prime2, sigma_t_prime2 = student_scheduler.get_alpha_sigma(
                            batch, timesteps // 2, accelerator.device
                        )
                        x_teacher_z_t_prime = (z_t - alpha_t_prime2 * rec_t_prime) / sigma_t_prime2
                        z_t_prime_2 = alpha_t_prime2 * x_teacher_z_t_prime - sigma_t_prime2 * rec_t_prime

                    noise_pred = student(z_t.permute(*permute_samples), timesteps).sample.permute(*permute_samples)
                    w = torch.pow(1 + alpha_t_prime2 / sigma_t_prime2, gamma)
                    loss = F.mse_loss(noise_pred * w, z_t_prime_2 * w)
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
                    new_scheduler = DDPMScheduler(
                        num_train_timesteps=n_teacher_trainsteps // 2,
                        beta_schedule="squaredcos_cap_v2",
                        variance_type="fixed_small_log",
                    )
                    pipeline = DDPMPipeline(
                        unet=accelerator.unwrap_model(ema_model.averaged_model if use_ema else student),
                        scheduler=new_scheduler,
                    )

                    # run pipeline in inference (sample random noise and denoise)
                    images = pipeline(batch_size=4, output_type="numpy", generator=torch.manual_seed(0)).images

                    # denormalize the images and save to tensorboard
                    images_processed = (images * 255).round().astype("uint8")
                    for sample_number, img in enumerate(images_processed):
                        img = Image.fromarray(img)

                        img.save(
                            os.path.join(
                                sample_path, f"{n_teacher_trainsteps}", f"epoch_{epoch}_sample_{sample_number}.png"
                            )
                        )
            accelerator.wait_for_everyone()
        return student, ema_model, accelerator
