import copy
import logging
import math
import os
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import torch
import transformers
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub.utils import insecure_hashlib
from peft import LoraConfig, prepare_model_for_kbit_training, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import crop
from tqdm.auto import tqdm

import diffusers
from diffusers import (
    AutoencoderKL, BitsAndBytesConfig, FlowMatchEulerDiscreteScheduler,
    FluxPipeline, FluxTransformer2DModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    cast_training_params, compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3, free_memory,
)
from diffusers.utils import convert_unet_state_dict_to_peft, is_wandb_available
from diffusers.utils.torch_utils import is_compiled_module

logger = get_logger(__name__)

class DreamBoothDataset(Dataset):
    def __init__(self, data_df_path, dataset_name, width, height, max_sequence_length=77):
        self.width, self.height, self.max_sequence_length = width, height, max_sequence_length
        self.data_df_path = Path(data_df_path)
        if not self.data_df_path.exists():
            raise ValueError("`data_df_path` doesn't exists.")

        dataset = load_dataset(dataset_name, split="train")
        self.instance_images = [sample["image"] for sample in dataset]
        self.image_hashes = [insecure_hashlib.sha256(img.tobytes()).hexdigest() for img in self.instance_images]
        self.pixel_values = self._apply_transforms()
        self.data_dict = self._map_embeddings()
        self._length = len(self.instance_images)

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        idx = index % len(self.instance_images)
        hash_key = self.image_hashes[idx]
        prompt_embeds, pooled_prompt_embeds, text_ids = self.data_dict[hash_key]
        return {
            "instance_images": self.pixel_values[idx],
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
            "text_ids": text_ids,
        }

    def _apply_transforms(self):
        transform = transforms.Compose([
            transforms.Resize((self.height, self.width), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomCrop((self.height, self.width)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        pixel_values = []
        for image in self.instance_images:
            image = exif_transpose(image).convert("RGB") if image.mode != "RGB" else exif_transpose(image)
            pixel_values.append(transform(image))
        return pixel_values

    def _map_embeddings(self):
        df = pd.read_parquet(self.data_df_path)
        data_dict = {}
        for _, row in df.iterrows():
            prompt_embeds = torch.from_numpy(np.array(row["prompt_embeds"]).reshape(self.max_sequence_length, 4096))
            pooled_prompt_embeds = torch.from_numpy(np.array(row["pooled_prompt_embeds"]).reshape(768))
            text_ids = torch.from_numpy(np.array(row["text_ids"]).reshape(77, 3))
            data_dict[row["image_hash"]] = (prompt_embeds, pooled_prompt_embeds, text_ids)
        return data_dict

def collate_fn(examples):
    pixel_values = torch.stack([ex["instance_images"] for ex in examples]).float()
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    prompt_embeds = torch.stack([ex["prompt_embeds"] for ex in examples])
    pooled_prompt_embeds = torch.stack([ex["pooled_prompt_embeds"] for ex in examples])
    text_ids = torch.stack([ex["text_ids"] for ex in examples])[0]

    return {
        "pixel_values": pixel_values,
        "prompt_embeds": prompt_embeds,
        "pooled_prompt_embeds": pooled_prompt_embeds,
        "text_ids": text_ids,
    }

def main(args):
    # Setup accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=ProjectConfiguration(project_dir=args.output_dir, logging_dir=Path(args.output_dir, "logs")),
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )

    # Setup logging
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    set_seed(args.seed) if args.seed is not None else None
    os.makedirs(args.output_dir, exist_ok=True) if accelerator.is_main_process else None

    # Load models with quantization
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)

    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")

    nf4_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    transformer = FluxTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="transformer",
        quantization_config=nf4_config, torch_dtype=torch.float16
    )
    transformer = prepare_model_for_kbit_training(transformer, use_gradient_checkpointing=False)

    # Freeze models and setup LoRA
    transformer.requires_grad_(False)
    vae.requires_grad_(False)
    vae.to(accelerator.device, dtype=torch.float16)
    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    # now we will add new LoRA weights to the attention layers
    transformer_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    transformer.add_adapter(transformer_lora_config)

    print(f"trainable params: {transformer.num_parameters(only_trainable=True)} || all params: {transformer.num_parameters()}")

    # Setup optimizer
    import bitsandbytes as bnb
    optimizer = bnb.optim.AdamW8bit(
        [{"params": list(filter(lambda p: p.requires_grad, transformer.parameters())), "lr": args.learning_rate}],
        betas=(0.9, 0.999), weight_decay=1e-04, eps=1e-08
    )

    # Setup dataset and dataloader
    train_dataset = DreamBoothDataset(args.data_df_path, "derekl35/alphonse-mucha-style", args.width, args.height)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn
    )

    # Cache latents
    vae_config = vae.config
    latents_cache = []
    for batch in tqdm(train_dataloader, desc="Caching latents"):
        with torch.no_grad():
            pixel_values = batch["pixel_values"].to(accelerator.device, dtype=torch.float16)
            latents_cache.append(vae.encode(pixel_values).latent_dist)

    del vae
    free_memory()

    # Setup scheduler and training steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    args.max_train_steps = args.max_train_steps or args.num_train_epochs * num_update_steps_per_epoch

    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler("constant", optimizer=optimizer, num_warmup_steps=0, num_training_steps=args.max_train_steps)

    # Prepare for training
    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(transformer, optimizer, train_dataloader, lr_scheduler)

    # Register save/load hooks
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        return model._orig_mod if is_compiled_module(model) else model

    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            for model in models:
                if isinstance(unwrap_model(model), type(unwrap_model(transformer))):
                    lora_layers = get_peft_model_state_dict(unwrap_model(model))
                    FluxPipeline.save_lora_weights(output_dir, transformer_lora_layers=lora_layers, text_encoder_lora_layers=None)
                    weights.pop() if weights else None

    accelerator.register_save_state_pre_hook(save_model_hook)
    cast_training_params([transformer], dtype=torch.float32) if args.mixed_precision == "fp16" else None

    # Initialize tracking
    accelerator.init_trackers("dreambooth-flux-dev-lora-alphonse-mucha", config=vars(args)) if accelerator.is_main_process else None

    # Training loop
    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps.to(accelerator.device)]
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    global_step = 0
    progress_bar = tqdm(range(args.max_train_steps), desc="Steps", disable=not accelerator.is_local_main_process)

    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate([transformer]):
                # Get cached latents
                model_input = latents_cache[step].sample()
                model_input = (model_input - vae_config.shift_factor) * vae_config.scaling_factor
                model_input = model_input.to(dtype=torch.float16)

                # Prepare inputs
                latent_image_ids = FluxPipeline._prepare_latent_image_ids(
                    model_input.shape[0], model_input.shape[2] // 2, model_input.shape[3] // 2,
                    accelerator.device, torch.float16
                )

                noise = torch.randn_like(model_input)
                bsz = model_input.shape[0]

                u = compute_density_for_timestep_sampling("none", bsz, 0.0, 1.0, 1.29)
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=model_input.device)

                sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
                noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise

                packed_noisy_model_input = FluxPipeline._pack_latents(
                    noisy_model_input, model_input.shape[0], model_input.shape[1],
                    model_input.shape[2], model_input.shape[3]
                )

                # Forward pass
                guidance = torch.tensor([args.guidance_scale], device=accelerator.device).expand(bsz) if unwrap_model(transformer).config.guidance_embeds else None

                model_pred = transformer(
                    hidden_states=packed_noisy_model_input,
                    timestep=timesteps / 1000,
                    guidance=guidance,
                    pooled_projections=batch["pooled_prompt_embeds"].to(accelerator.device, dtype=torch.float16),
                    encoder_hidden_states=batch["prompt_embeds"].to(accelerator.device, dtype=torch.float16),
                    txt_ids=batch["text_ids"].to(accelerator.device, dtype=torch.float16),
                    img_ids=latent_image_ids,
                    return_dict=False,
                )[0]

                vae_scale_factor = 2 ** (len(vae_config.block_out_channels) - 1)
                model_pred = FluxPipeline._unpack_latents(
                    model_pred, model_input.shape[2] * vae_scale_factor,
                    model_input.shape[3] * vae_scale_factor, vae_scale_factor
                )

                # Compute loss
                weighting = compute_loss_weighting_for_sd3("none", sigmas)
                target = noise - model_input
                loss = torch.mean((weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1), 1).mean()

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(transformer.parameters(), 1.0)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                # Checkpointing
                if global_step % args.checkpointing_steps == 0 and (accelerator.is_main_process or accelerator.distributed_type == DistributedType.DEEPSPEED):
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)

            # Logging
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # Final save
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        transformer_lora_layers = get_peft_model_state_dict(unwrap_model(transformer))
        FluxPipeline.save_lora_weights(args.output_dir, transformer_lora_layers=transformer_lora_layers, text_encoder_lora_layers=None)

    if torch.cuda.is_available():
        print(f"Pipeline memory usage: {torch.cuda.max_memory_reserved() / 1024**3:.3f} GB")
    else:
        print("Training completed. GPU not available for memory tracking.")

    accelerator.end_training()

if __name__ == "__main__":
    class Args:
        pretrained_model_name_or_path = "black-forest-labs/FLUX.1-dev"
        data_df_path = "embeddings_alphonse_mucha.parquet"
        output_dir = "alphonse_mucha_lora_flux_nf4"
        mixed_precision = "fp16"
        weighting_scheme = "none"
        width, height = 512, 768
        train_batch_size = 1
        learning_rate = 1e-4
        guidance_scale = 1.0
        report_to = "wandb"
        gradient_accumulation_steps = 4
        gradient_checkpointing = True
        rank = 4
        max_train_steps = 700
        seed = 0
        checkpointing_steps = 100

    main(Args())