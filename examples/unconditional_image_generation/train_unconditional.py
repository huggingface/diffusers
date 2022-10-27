import argparse
import math
import os

import torch
import torch.nn.functional as F

from accelerate import Accelerator
from accelerate.logging import get_logger
from datasets import load_dataset
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel
from diffusers.hub_utils import init_git_repo, push_to_hub
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from torchvision.transforms import (
    CenterCrop,
    Compose,
    InterpolationMode,
    Normalize,
    RandomHorizontalFlip,
    Resize,
    ToTensor,
)
from tqdm.auto import tqdm


logger = get_logger(__name__)


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    if not isinstance(arr, torch.Tensor):
        arr = torch.from_numpy(arr)
    res = arr[timesteps].float().to(timesteps.device)
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


def main(args):
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        logging_dir=logging_dir,
    )

    model = UNet2DModel(
        sample_size=args.resolution,
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
    noise_scheduler = DDPMScheduler(num_train_timesteps=args.ddpm_num_steps,
                                    beta_schedule=args.ddpm_beta_schedule,
                                    tensor_format="pt")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    augmentations = Compose(
        [
            Resize(args.resolution, interpolation=InterpolationMode.BILINEAR),
            CenterCrop(args.resolution),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize([0.5], [0.5]),
        ]
    )

    if args.dataset_name is not None:
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
            use_auth_token=True if args.use_auth_token else None,
            split="train",
        )
    else:
        dataset = load_dataset("imagefolder", data_dir=args.train_data_dir, cache_dir=args.cache_dir, split="train")

    def transforms(examples):
        images = [augmentations(image.convert("RGB")) for image in examples["image"]]
        return {"input": images}

    logger.info(f"Length of dataset: {len(dataset)}")

    dataset.set_transform(transforms)
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True)

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * args.num_epochs) // args.gradient_accumulation_steps,
    )

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)

    ema_model = EMAModel(model, inv_gamma=args.ema_inv_gamma, power=args.ema_power, max_value=args.ema_max_decay)

    if args.push_to_hub:
        repo = init_git_repo(args, at_init=True)

    if accelerator.is_main_process:
        run = os.path.split(__file__)[-1].split(".")[0]
        accelerator.init_trackers(run)

    global_step = 0
    for epoch in range(args.num_epochs):
        model.train()
        progress_bar = tqdm(total=num_update_steps_per_epoch, disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in enumerate(train_dataloader):
            clean_images = batch["input"]
            # Sample noise that we'll add to the images
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bsz = clean_images.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bsz,), device=clean_images.device
            ).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                # Predict the noise residual
                model_output = model(noisy_images, timesteps).sample

                if args.predict_mode == "eps":
                    loss = F.mse_loss(model_output, noise)      # this could have different weights!
                elif args.predict_mode == "x0":
                    alpha_t = _extract_into_tensor(noise_scheduler.alphas_cumprod, timesteps, clean_images[:,0:1,0:1,0:1].shape)
                    snr_weights = alpha_t / (1 - alpha_t)
                    loss = snr_weights * F.mse_loss(model_output, clean_images, reduction="none")   # use SNR weighting from distillation paper
                    loss = loss.mean()

                accelerator.backward(loss)

                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                if args.use_ema:
                    ema_model.step(model)
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            if args.use_ema:
                logs["ema_decay"] = ema_model.decay
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
        progress_bar.close()

        accelerator.wait_for_everyone()

        # Generate sample images for visual inspection
        if accelerator.is_main_process:
            if epoch % args.save_images_epochs == 0 or epoch == args.num_epochs - 1:

                pipeline = DDPMPipeline(
                    unet=accelerator.unwrap_model(ema_model.averaged_model if args.use_ema else model),
                    scheduler=noise_scheduler,
                )

                generator = torch.manual_seed(0)
                # run pipeline in inference (sample random noise and denoise)
                images = pipeline(generator=generator, batch_size=args.eval_batch_size, output_type="numpy",
                    predict_epsilon=args.predict_mode == "eps",).images

                # denormalize the images and save to tensorboard
                images_processed = (images * 255).round().astype("uint8")
                accelerator.trackers[0].writer.add_images(
                    "test_samples", images_processed.transpose(0, 3, 1, 2), epoch
                )

            if epoch % args.save_model_epochs == 0 or epoch == args.num_epochs - 1:
                # save the model
                if args.push_to_hub:
                    push_to_hub(args, pipeline, repo, commit_message=f"Epoch {epoch}", blocking=False)
                else:
                    pipeline.save_pretrained(args.output_dir)
        accelerator.wait_for_everyone()

    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--dataset_config_name", type=str, default=None)
    parser.add_argument("--predict_mode", type=str, default="eps",
                        help="What the model should predict. 'eps' to predict error, 'x0' to directly predict reconstruction")
    parser.add_argument("--train_data_dir", type=str, default=None, help="A folder containing the training data.")
    parser.add_argument("--ddpm_num_steps", type=int, default=1000)
    parser.add_argument("--ddpm_beta_schedule", type=str, default="linear")
    parser.add_argument("--output_dir", type=str, default="ddpm-model-64")
    parser.add_argument("--overwrite_output_dir", action="store_true")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--save_images_epochs", type=int, default=10)
    parser.add_argument("--save_model_epochs", type=int, default=10)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler", type=str, default="cosine")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--adam_beta1", type=float, default=0.95)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-6)
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    parser.add_argument("--use_ema", action="store_true", default=True)
    parser.add_argument("--ema_inv_gamma", type=float, default=1.0)
    parser.add_argument("--ema_power", type=float, default=3 / 4)
    parser.add_argument("--ema_max_decay", type=float, default=0.9999)
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--use_auth_token", action="store_true")
    parser.add_argument("--hub_token", type=str, default=None)
    parser.add_argument("--hub_model_id", type=str, default=None)
    parser.add_argument("--hub_private_repo", action="store_true")
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("You must specify either a dataset name from the hub or a train data directory.")

    main(args)
