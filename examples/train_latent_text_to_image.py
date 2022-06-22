import argparse
import os

import torch
import torch.nn.functional as F

import PIL.Image
from accelerate import Accelerator
from datasets import load_dataset
from diffusers import DDPM, DDPMScheduler, UNetLDMModel
from diffusers.hub_utils import init_git_repo, push_to_hub
from diffusers.modeling_utils import unwrap_model
from diffusers.optimization import get_scheduler
from diffusers.utils import logging
from torchvision.transforms import (
    CenterCrop,
    Compose,
    InterpolationMode,
    Lambda,
    RandomHorizontalFlip,
    Resize,
    ToTensor,
)
from tqdm.auto import tqdm


logger = logging.get_logger(__name__)


def main(args):
    accelerator = Accelerator(mixed_precision=args.mixed_precision)

    model = UNetLDMModel(
        attention_resolutions=[4, 2, 1],
        channel_mult=[1, 2, 4, 4],
        context_dim=1280,
        conv_resample=True,
        dims=2,
        dropout=0,
        image_size=32,
        in_channels=4,
        model_channels=320,
        num_heads=8,
        num_res_blocks=2,
        out_channels=4,
        resblock_updown=False,
        transformer_depth=1,
        use_new_attention_order=False,
        use_scale_shift_norm=False,
        use_spatial_transformer=True,
        legacy=False,
    )
    noise_scheduler = DDPMScheduler(timesteps=1000, tensor_format="pt")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    augmentations = Compose(
        [
            Resize(args.resolution, interpolation=InterpolationMode.BILINEAR),
            CenterCrop(args.resolution),
            RandomHorizontalFlip(),
            ToTensor(),
            Lambda(lambda x: x * 2 - 1),
        ]
    )
    dataset = load_dataset(args.dataset, split="train")

    def transforms(examples):
        images = [augmentations(image.convert("RGB")) for image in examples["image"]]
        return {"input": images}

    dataset.set_transform(transforms)
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=(len(train_dataloader) * args.num_epochs) // args.gradient_accumulation_steps,
    )

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    if args.push_to_hub:
        repo = init_git_repo(args, at_init=True)

    # Train!
    is_distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    world_size = torch.distributed.get_world_size() if is_distributed else 1
    total_train_batch_size = args.batch_size * args.gradient_accumulation_steps * world_size
    max_steps = len(train_dataloader) // args.gradient_accumulation_steps * args.num_epochs
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataloader.dataset)}")
    logger.info(f"  Num Epochs = {args.num_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_steps}")

    for epoch in range(args.num_epochs):
        model.train()
        with tqdm(total=len(train_dataloader), unit="ba") as pbar:
            pbar.set_description(f"Epoch {epoch}")
            for step, batch in enumerate(train_dataloader):
                clean_images = batch["input"]
                noise_samples = torch.randn(clean_images.shape).to(clean_images.device)
                bsz = clean_images.shape[0]
                timesteps = torch.randint(0, noise_scheduler.timesteps, (bsz,), device=clean_images.device).long()

                # add noise onto the clean images according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_images = noise_scheduler.training_step(clean_images, noise_samples, timesteps)

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
                    optimizer.zero_grad()
                pbar.update(1)
                pbar.set_postfix(loss=loss.detach().item(), lr=optimizer.param_groups[0]["lr"])

                optimizer.step()
        if is_distributed:
            torch.distributed.barrier()

        # Generate a sample image for visual inspection
        if args.local_rank in [-1, 0]:
            model.eval()
            with torch.no_grad():
                pipeline = DDPM(unet=unwrap_model(model), noise_scheduler=noise_scheduler)

                generator = torch.manual_seed(0)
                # run pipeline in inference (sample random noise and denoise)
                image = pipeline(generator=generator)

            # process image to PIL
            image_processed = image.cpu().permute(0, 2, 3, 1)
            image_processed = (image_processed + 1.0) * 127.5
            image_processed = image_processed.type(torch.uint8).numpy()
            image_pil = PIL.Image.fromarray(image_processed[0])

            # save image
            test_dir = os.path.join(args.output_dir, "test_samples")
            os.makedirs(test_dir, exist_ok=True)
            image_pil.save(f"{test_dir}/{epoch:04d}.png")

            # save the model
            if args.push_to_hub:
                push_to_hub(args, pipeline, repo, commit_message=f"Epoch {epoch}", blocking=False)
            else:
                pipeline.save_pretrained(args.output_dir)
        if is_distributed:
            torch.distributed.barrier()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--dataset", type=str, default="huggan/flowers-102-categories")
    parser.add_argument("--output_dir", type=str, default="ddpm-model")
    parser.add_argument("--overwrite_output_dir", action="store_true")
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_token", type=str, default=None)
    parser.add_argument("--hub_model_id", type=str, default=None)
    parser.add_argument("--hub_private_repo", action="store_true")
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
