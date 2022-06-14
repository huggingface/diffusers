import os

import torch
import PIL.Image
import argparse
import torch.nn.functional as F

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
        resolution=64,
    )
    noise_scheduler = DDPMScheduler(timesteps=1000)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 100
    batch_size = 16
    gradient_accumulation_steps = 1

    augmentations = Compose(
        [
            Resize(64, interpolation=InterpolationMode.BILINEAR),
            RandomCrop(64),
            RandomHorizontalFlip(),
            ToTensor(),
            Lambda(lambda x: x * 2 - 1),
        ]
    )
    dataset = load_dataset("huggan/pokemon", split="train")

    def transforms(examples):
        images = [augmentations(image.convert("RGB")) for image in examples["image"]]
        return {"input": images}

    dataset.set_transform(transforms)
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=(len(train_dataloader) * num_epochs) // gradient_accumulation_steps,
    )

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    for epoch in range(num_epochs):
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

                if step % gradient_accumulation_steps != 0:
                    with accelerator.no_sync(model):
                        output = model(noisy_images, timesteps)
                        # predict the noise
                        loss = F.mse_loss(output, noise_samples)
                        accelerator.backward(loss)
                else:
                    output = model(noisy_images, timesteps)
                    loss = F.mse_loss(output, noise_samples)
                    accelerator.backward(loss)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                pbar.update(1)
                pbar.set_postfix(loss=loss.detach().item(), lr=optimizer.param_groups[0]["lr"])

                optimizer.step()

        torch.distributed.barrier()
        if args.local_rank in [-1, 0]:
            model.eval()
            with torch.no_grad():
                pipeline = DDPM(unet=model.module, noise_scheduler=noise_scheduler)
                generator = torch.Generator()
                generator = generator.manual_seed(0)
                # run pipeline in inference (sample random noise and denoise)
                image = pipeline(generator=generator)

                # process image to PIL
                image_processed = image.cpu().permute(0, 2, 3, 1)
                image_processed = (image_processed + 1.0) * 127.5
                image_processed = image_processed.type(torch.uint8).numpy()
                image_pil = PIL.Image.fromarray(image_processed[0])

                # save image
                pipeline.save_pretrained("./pokemon-ddpm")
                image_pil.save(f"./pokemon-ddpm/test_{epoch}.png")
        torch.distributed.barrier()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple example of training script.")
    parser.add_argument("--local_rank", type=int)
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
