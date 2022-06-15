import argparse
import os

import torch
import torch.nn.functional as F

import PIL.Image
from accelerate import Accelerator
from datasets import load_dataset
from diffusers import DDPM, DDPMScheduler, UNetModel
from torchvision.transforms import (
    Compose,
    InterpolationMode,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip,
    Resize,
    ToTensor,
)
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup


def main(args):
    accelerator = Accelerator(mixed_precision=args.mixed_precision)

    model = UNetModel(
        attn_resolutions=(16,),
        ch=128,
        ch_mult=(1, 2, 4, 8),
        dropout=0.0,
        num_res_blocks=2,
        resamp_with_conv=True,
        resolution=args.resolution,
    )
    noise_scheduler = DDPMScheduler(timesteps=1000)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    augmentations = Compose(
        [
            Resize(args.resolution, interpolation=InterpolationMode.BILINEAR),
            RandomCrop(args.resolution),
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

    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=(len(train_dataloader) * args.num_epochs) // args.gradient_accumulation_steps,
    )

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    for epoch in range(args.num_epochs):
        model.train()
        with tqdm(total=len(train_dataloader), unit="ba") as pbar:
            pbar.set_description(f"Epoch {epoch}")
            for step, batch in enumerate(train_dataloader):
                clean_images = batch["input"]
                noisy_images = torch.empty_like(clean_images)
                noise_samples = torch.empty_like(clean_images)
                bsz = clean_images.shape[0]

                timesteps = torch.randint(0, noise_scheduler.timesteps, (bsz,), device=clean_images.device).long()
                for idx in range(bsz):
                    noise = torch.randn(clean_images.shape[1:]).to(clean_images.device)
                    noise_samples[idx] = noise
                    noisy_images[idx] = noise_scheduler.forward_step(clean_images[idx], noise, timesteps[idx])

                if step % args.gradient_accumulation_steps != 0:
                    with accelerator.no_sync(model):
                        output = model(noisy_images, timesteps)
                        # predict the noise residual
                        loss = F.mse_loss(output, noise_samples)
                        accelerator.backward(loss)
                else:
                    output = model(noisy_images, timesteps)
                    # predict the noise residual
                    loss = F.mse_loss(output, noise_samples)
                    accelerator.backward(loss)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                pbar.update(1)
                pbar.set_postfix(loss=loss.detach().item(), lr=optimizer.param_groups[0]["lr"])

                optimizer.step()

        # Generate a sample image for visual inspection
        torch.distributed.barrier()
        if args.local_rank in [-1, 0]:
            model.eval()
            with torch.no_grad():
                if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                    pipeline = DDPM(unet=model.module, noise_scheduler=noise_scheduler)
                else:
                    pipeline = DDPM(unet=model, noise_scheduler=noise_scheduler)
                pipeline.save_pretrained(args.output_path)

                generator = torch.manual_seed(0)
                # run pipeline in inference (sample random noise and denoise)
                image = pipeline(generator=generator)

                # process image to PIL
                image_processed = image.cpu().permute(0, 2, 3, 1)
                image_processed = (image_processed + 1.0) * 127.5
                image_processed = image_processed.type(torch.uint8).numpy()
                image_pil = PIL.Image.fromarray(image_processed[0])

                # save image
                test_dir = os.path.join(args.output_path, "test_samples")
                os.makedirs(test_dir, exist_ok=True)
                image_pil.save(f"{test_dir}/{epoch}.png")
        torch.distributed.barrier()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--dataset", type=str, default="huggan/flowers-102-categories")
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--output_path", type=str, default="ddpm-model")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help="Whether to use mixed precision. Choose"
        "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
        "and an Nvidia Ampere GPU.",
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    main(args)
