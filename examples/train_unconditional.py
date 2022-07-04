import argparse
import os

import torch
import torch.nn.functional as F

from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.logging import get_logger
from datasets import load_dataset
from diffusers import DDIMPipeline, DDIMScheduler, UNetModel
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


def main(args):
    ddp_unused_params = DistributedDataParallelKwargs(find_unused_parameters=True)
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        logging_dir=logging_dir,
        kwargs_handlers=[ddp_unused_params],
    )

    model = UNetModel(
        attn_resolutions=(16,),
        ch=128,
        ch_mult=(1, 2, 4, 8),
        dropout=0.0,
        num_res_blocks=2,
        resamp_with_conv=True,
        resolution=args.resolution,
    )
    noise_scheduler = DDIMScheduler(timesteps=1000, tensor_format="pt")
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
    dataset = load_dataset(args.dataset, split="train")

    def transforms(examples):
        images = [augmentations(image.convert("RGB")) for image in examples["image"]]
        return {"input": images}

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

    ema_model = EMAModel(model, inv_gamma=args.ema_inv_gamma, power=args.ema_power, max_value=args.ema_max_decay)

    if args.push_to_hub:
        repo = init_git_repo(args, at_init=True)

    if accelerator.is_main_process:
        run = os.path.split(__file__)[-1].split(".")[0]
        accelerator.init_trackers(run)

    # Train!
    is_distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    world_size = torch.distributed.get_world_size() if is_distributed else 1
    total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * world_size
    max_steps = len(train_dataloader) // args.gradient_accumulation_steps * args.num_epochs
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataloader.dataset)}")
    logger.info(f"  Num Epochs = {args.num_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_steps}")

    global_step = 0
    for epoch in range(args.num_epochs):
        model.train()
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in enumerate(train_dataloader):
            clean_images = batch["input"]
            noise_samples = torch.randn(clean_images.shape).to(clean_images.device)
            bsz = clean_images.shape[0]
            timesteps = torch.randint(0, noise_scheduler.timesteps, (bsz,), device=clean_images.device).long()

            # add noise onto the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise_samples, timesteps)

            if step % args.gradient_accumulation_steps != 0:
                with accelerator.no_sync(model):
                    output = model(noisy_images, timesteps)
                    # predict the noise residual
                    loss = F.mse_loss(output, noise_samples)
                    loss = loss / args.gradient_accumulation_steps
                    accelerator.backward(loss)
            else:
                output = model(noisy_images, timesteps)
                # predict the noise residual
                loss = F.mse_loss(output, noise_samples)
                loss = loss / args.gradient_accumulation_steps
                accelerator.backward(loss)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                ema_model.step(model)
                optimizer.zero_grad()
            progress_bar.update(1)
            progress_bar.set_postfix(
                loss=loss.detach().item(), lr=optimizer.param_groups[0]["lr"], ema_decay=ema_model.decay
            )
            accelerator.log(
                {
                    "train_loss": loss.detach().item(),
                    "epoch": epoch,
                    "ema_decay": ema_model.decay,
                    "step": global_step,
                },
                step=global_step,
            )
            global_step += 1
        progress_bar.close()

        accelerator.wait_for_everyone()

        # Generate a sample image for visual inspection
        if accelerator.is_main_process:
            with torch.no_grad():
                pipeline = DDIMPipeline(
                    unet=accelerator.unwrap_model(ema_model.averaged_model),
                    noise_scheduler=noise_scheduler,
                )

                generator = torch.manual_seed(0)
                # run pipeline in inference (sample random noise and denoise)
                images = pipeline(generator=generator, batch_size=args.eval_batch_size, num_inference_steps=50)

            # denormalize the images and save to tensorboard
            images_processed = (images.cpu() + 1.0) * 127.5
            images_processed = images_processed.clamp(0, 255).type(torch.uint8).numpy()

            accelerator.trackers[0].writer.add_images("test_samples", images_processed, epoch)

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
    parser.add_argument("--dataset", type=str, default="huggan/flowers-102-categories")
    parser.add_argument("--output_dir", type=str, default="ddpm-model")
    parser.add_argument("--overwrite_output_dir", action="store_true")
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler", type=str, default="cosine")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--adam_beta1", type=float, default=0.95)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-6)
    parser.add_argument("--adam_epsilon", type=float, default=1e-3)
    parser.add_argument("--ema_inv_gamma", type=float, default=1.0)
    parser.add_argument("--ema_power", type=float, default=3 / 4)
    parser.add_argument("--ema_max_decay", type=float, default=0.9999)
    parser.add_argument("--push_to_hub", action="store_true")
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

    main(args)
