import argparse
import copy
import itertools
import logging
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import concatenate_datasets, load_dataset
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionInpaintPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available


if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.13.0.dev0")

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument("--instance_data_dir", nargs="+", help="Instance data directories")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="text-inversion-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--train_text_encoder", default=False, action="store_true", help="Whether to train the text encoder"
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--sample_batch_size", type=int, default=4, help="Batch size (per device) for sampling images."
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
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
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=1000,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint and are suitable for resuming training"
            " using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpointing_from",
        type=int,
        default=1000,
        help=("Start to checkpoint from step"),
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=50,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument(
        "--validation_from",
        type=int,
        default=0,
        help=("Start to validate from step"),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=(
            "Max number of checkpoints to store. Passed as `total_limit` to the `Accelerator` `ProjectConfiguration`."
            " See Accelerator::save_state https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.save_state"
            " for more docs"
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--validation_project_name",
        type=str,
        default=None,
        help="The w&b name.",
    )
    parser.add_argument(
        "--report_to_wandb", default=False, action="store_true", help="Whether to report to weights and biases"
    )

    args = parser.parse_args()

    return args


def prepare_mask_and_masked_image(image, mask):
    image = np.array(image.convert("RGB"))
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

    mask = np.array(mask.convert("L"))
    mask = mask.astype(np.float32) / 255.0
    mask = mask[None, None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)

    masked_image = image * (mask < 0.5)

    return mask, masked_image


class DreamBoothDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        datasets_paths,
    ):
        self.tokenizer = tokenizer
        self.datasets_paths = (datasets_paths,)
        self.datasets = [load_dataset(dataset_path) for dataset_path in self.datasets_paths[0]]
        self.train_data = concatenate_datasets([dataset["train"] for dataset in self.datasets])
        self.test_data = concatenate_datasets([dataset["test"] for dataset in self.datasets])

        self.image_normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def set_image(self, img, switch):
        if img.mode not in ["RGB", "L"]:
            img = img.convert("RGB")

        if switch:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        img = img.resize((512, 512), Image.BILINEAR)

        return img

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, index):
        # Lettings
        example = {}
        img_idx = index % len(self.train_data)
        switch = random.choice([True, False])

        # Load image
        image = self.set_image(self.train_data[img_idx]["image"], switch)

        # Normalize image
        image_norm = self.image_normalize(image)

        # Tokenise prompt
        tokenized_prompt = self.tokenizer(
            self.train_data[img_idx]["prompt"],
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids

        # Load masks for image
        masks = [
            self.set_image(self.train_data[img_idx][key], switch) for key in self.train_data[img_idx] if "mask" in key
        ]

        # Build example
        example["PIL_image"] = image
        example["instance_image"] = image_norm
        example["instance_prompt_id"] = tokenized_prompt
        example["instance_masks"] = masks

        return example


def weighted_mask(masks):
    # Convert each mask to a NumPy array and ensure it's binary
    mask_arrays = [np.array(mask) / 255 for mask in masks]  # Normalizing to 0-1 range

    # Generate random weights and apply them to each mask
    weights = [random.random() for _ in masks]
    weights = [weight / sum(weights) for weight in weights]
    weighted_masks = [mask * weight for mask, weight in zip(mask_arrays, weights)]

    # Sum the weighted masks
    summed_mask = np.sum(weighted_masks, axis=0)

    # Apply a threshold to create the final mask
    threshold = 0.5  # This threshold can be adjusted
    result_mask = summed_mask >= threshold

    # Convert the result back to a PIL image
    return Image.fromarray(result_mask.astype(np.uint8) * 255)


def collate_fn(examples, tokenizer):
    input_ids = [example["instance_prompt_id"] for example in examples]
    pixel_values = [example["instance_image"] for example in examples]

    masks, masked_images = [], []

    for example in examples:
        # generate a random mask
        mask = weighted_mask(example["instance_masks"])

        # prepare mask and masked image
        mask, masked_image = prepare_mask_and_masked_image(example["PIL_image"], mask)

        masks.append(mask)
        masked_images.append(masked_image)

    pixel_values = torch.stack(pixel_values).to(memory_format=torch.contiguous_format).float()
    masks = torch.stack(masks)
    masked_images = torch.stack(masked_images)
    input_ids = tokenizer.pad({"input_ids": input_ids}, padding=True, return_tensors="pt").input_ids

    batch = {"input_ids": input_ids, "pixel_values": pixel_values, "masks": masks, "masked_images": masked_images}

    return batch


def log_validation(pipeline, text_encoder, unet, val_pairs, accelerator):
    # update pipeline (note: unet and vae are loaded again in float32)
    pipeline.text_encoder = accelerator.unwrap_model(text_encoder)
    pipeline.unet = accelerator.unwrap_model(unet)

    with torch.autocast("cuda"):
        val_results = [{"data_or_path": pipeline(**pair).images[0], "caption": pair["prompt"]} for pair in val_pairs]

    torch.cuda.empty_cache()

    wandb.log({"validation": [wandb.Image(**val_result) for val_result in val_results]})


def checkpoint(args, global_step, accelerator):
    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
    accelerator.save_state(save_path)
    logger.info(f"Saved state to {save_path}")


def main():
    args = parse_args()

    project_config = ProjectConfiguration(
        total_limit=args.checkpoints_total_limit,
        project_dir=args.output_dir,
        logging_dir=Path(args.output_dir, args.logging_dir),
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_config=project_config,
        log_with="wandb" if args.report_to_wandb else None,
    )

    if args.report_to_wandb and not is_wandb_available():
        raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    if args.seed is not None:
        set_seed(args.seed)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    # Load the tokenizer & models and create wrapper for stable diffusion
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder"
    ).requires_grad_(args.train_text_encoder)
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae").requires_grad_(False)
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    optimizer = torch.optim.AdamW(
        params=itertools.chain(unet.parameters(), text_encoder.parameters())
        if args.train_text_encoder
        else unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    train_dataset = DreamBoothDataset(
        tokenizer=tokenizer,
        datasets_paths=args.instance_data_dir,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=lambda examples: collate_fn(examples, tokenizer),
    )

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    if args.train_text_encoder:
        unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, text_encoder, optimizer, train_dataloader, lr_scheduler
        )
    else:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )

    accelerator.register_for_checkpointing(lr_scheduler)

    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    else:
        weight_dtype = torch.float32

    # Move text_encode and vae to gpu.
    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    vae.to(accelerator.device, dtype=weight_dtype)
    if not args.train_text_encoder:
        text_encoder.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)

    # Afterwards we calculate our number of training epochs
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = vars(copy.deepcopy(args))
        accelerator.init_trackers(args.validation_project_name, config=tracker_config)

    # create validation pipeline (note: unet and vae are loaded again in float32)
    val_pipeline = StableDiffusionInpaintPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        unet=unet,
        vae=vae,
        torch_dtype=weight_dtype,
        safety_checker=None,
    )
    val_pipeline.set_progress_bar_config(disable=True)

    # prepare validation dataset
    val_pairs = [
        {
            "image": example["image"],
            "mask_image": mask,
            "prompt": example["prompt"],
        }
        for example in train_dataset.test_data
        for mask in [example[key] for key in example if "mask" in key]
    ]

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            for model in models:
                sub_dir = "unet" if isinstance(model, type(accelerator.unwrap_model(unet))) else "text_encoder"
                model.save_pretrained(os.path.join(output_dir, sub_dir))

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

    accelerator.register_save_state_pre_hook(save_model_hook)

    print()

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    global_step = 0
    first_epoch = 0

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, num_train_epochs):
        unet.train()
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(unet):
                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Convert masked images to latent space
                masked_latents = vae.encode(
                    batch["masked_images"].reshape(batch["pixel_values"].shape).to(dtype=weight_dtype)
                ).latent_dist.sample()
                masked_latents = masked_latents * vae.config.scaling_factor

                masks = batch["masks"]
                # resize the mask to latents shape as we concatenate the mask to the latents
                mask = torch.stack(
                    [
                        torch.nn.functional.interpolate(mask, size=(args.resolution // 8, args.resolution // 8))
                        for mask in masks
                    ]
                )
                mask = mask.reshape(-1, 1, args.resolution // 8, args.resolution // 8)

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # concatenate the noised latents with the mask and the masked latents
                latent_model_input = torch.cat([noisy_latents, mask, masked_latents], dim=1)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                # Predict the noise residual
                noise_pred = unet(latent_model_input, timesteps, encoder_hidden_states).sample

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                loss = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = (
                        itertools.chain(unet.parameters(), text_encoder.parameters())
                        if args.train_text_encoder
                        else unet.parameters()
                    )
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if (
                        global_step % args.validation_steps == 0
                        and global_step >= args.validation_from
                        and args.report_to_wandb
                    ):
                        log_validation(
                            val_pipeline,
                            text_encoder,
                            unet,
                            val_pairs,
                            accelerator,
                        )

                    if global_step % args.checkpointing_steps == 0 and global_step >= args.checkpointing_from:
                        checkpoint(
                            args,
                            global_step,
                            accelerator,
                        )

            # Step logging
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

        accelerator.wait_for_everyone()

    # Terminate training
    accelerator.end_training()


if __name__ == "__main__":
    main()
