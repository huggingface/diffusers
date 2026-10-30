#!/usr/bin/env python
# coding=utf-8
# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""LoRA fine-tuning of the RFD4-Proteina denoiser on a folder of structures."""

import argparse
import logging
import os
import shutil
from pathlib import Path

import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

import diffusers
from diffusers import RFD4ProteinaAutoencoder, RFD4ProteinaPipeline, RFD4ProteinaScheduler, RFD4ProteinaTransformer
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params
from diffusers.utils import convert_state_dict_to_diffusers, convert_unet_state_dict_to_peft


logger = get_logger(__name__)

# The pipeline takes the molecular tensors by name and the rest of the prepared features as conditioning.
PIPELINE_INPUTS = (
    "padding_mask",
    "is_in_context_conditioning",
    "supports_local_latent",
    "token_type",
    "atom37_mask",
    "coords_nm",
)

# Training times per modality, as the released recipe samples them: a Beta distribution mixed with a uniform one.
# See `loss.t_distribution` in the RFD4-Proteina `configs/training_proteina.yaml`.
TIME_DISTRIBUTION = {
    "bb_ca": {"concentration1": 1.9, "concentration0": 1.0, "uniform_fraction": 0.02},
    "local_latents": {"concentration1": 1.0, "concentration0": 1.5, "uniform_fraction": 0.2},
}

# The backbone is the modality the sampler draws with zero centre of mass and a per-molecule stochastic offset.
BACKBONE = "bb_ca"


def parse_args():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for RFD4-Proteina.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        required=True,
        help="Repository id or local path of the RFD4-Proteina pipeline to adapt.",
    )
    parser.add_argument("--revision", type=str, default=None, help="Model revision to load.")
    parser.add_argument("--variant", type=str, default=None, help="Weight variant to load, such as `fp16`.")
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        required=True,
        help="Directory searched recursively for the annotated CIF files to fine-tune on.",
    )
    parser.add_argument(
        "--features_cache_dir",
        type=str,
        default=None,
        help="Directory for the prepared tensors. Preparation runs once per file and is reused across epochs.",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=384,
        help="Skip structures with more tokens than this, as the released fine-tune does.",
    )
    parser.add_argument("--prep_seed", type=int, default=0, help="Seed for the AtomWorks preparation pipeline.")
    parser.add_argument("--output_dir", type=str, default="rfd4-proteina-lora", help="Where the adapter is written.")
    parser.add_argument("--cache_dir", type=str, default=None, help="Hugging Face cache directory.")
    parser.add_argument("--seed", type=int, default=None, help="Seed for training.")
    parser.add_argument("--rank", type=int, default=32, help="LoRA rank.")
    parser.add_argument("--lora_alpha", type=float, default=64.0, help="LoRA scaling factor.")
    parser.add_argument("--lora_dropout", type=float, default=0.0, help="Dropout on the LoRA layers.")
    parser.add_argument(
        "--target_modules",
        type=str,
        nargs="+",
        default=["to_q", "to_k", "to_v", "to_out"],
        help="Attention projections to adapt. The default is the released recipe's `to_qkv` and `to_out_node`.",
    )
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Initial learning rate.")
    parser.add_argument("--lr_scheduler", type=str, default="constant", help="Learning rate schedule.")
    parser.add_argument("--lr_warmup_steps", type=int, default=0, help="Warmup steps for the learning rate.")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="Adam beta1.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="Adam beta2.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-4, help="Weight decay for the adapters.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8, help="Adam epsilon.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Gradient clipping norm.")
    parser.add_argument("--max_train_steps", type=int, default=500, help="Optimizer steps to run.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="Batches per optimizer step.")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Trade compute for memory.")
    parser.add_argument(
        "--recycle_depth",
        type=int,
        default=1,
        help="Trunk passes per training step. Sampling recycles more deeply and more slowly; raising this costs "
        "proportionally more time and memory per step.",
    )
    parser.add_argument("--dataloader_num_workers", type=int, default=0, help="Worker processes for the loader.")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help="Precision of the frozen base model. The adapters always train in float32.",
    )
    parser.add_argument("--checkpointing_steps", type=int, default=250, help="Steps between training checkpoints.")
    parser.add_argument("--checkpoints_total_limit", type=int, default=None, help="Checkpoints to keep.")
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="A checkpoint directory under `output_dir`, or `latest` for the most recent one.",
    )
    parser.add_argument("--logging_dir", type=str, default="logs", help="Log directory under `output_dir`.")
    parser.add_argument("--report_to", type=str, default="tensorboard", help="Tracker to log to.")
    parser.add_argument(
        "--validation_cif",
        type=str,
        default=None,
        help="Annotated CIF generated from after every `validation_steps` to watch the adapter's effect.",
    )
    parser.add_argument("--validation_steps", type=int, default=250, help="Steps between validation generations.")
    return parser.parse_args()


class PreparedStructureDataset(Dataset):
    """The CIF files under a directory, prepared into the feature tensors the model consumes.

    Preparation parses the file with AtomWorks and runs the model's own datapipe, which is slow enough to be worth
    caching: each file is prepared once and reloaded from `cache_dir` afterwards.
    """

    def __init__(self, data_dir: str, seed: int, max_tokens: int, cache_dir: str | None = None):
        from rfd4_proteina_prep import prepare_cif

        self._prepare_cif = prepare_cif
        self.seed = seed
        self.max_tokens = max_tokens
        self.cache_dir = None if cache_dir is None else Path(cache_dir)
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        root = Path(data_dir)
        if not root.is_dir():
            raise ValueError(f"{data_dir} is not a directory.")
        self.paths = sorted(path for pattern in ("*.cif", "*.cif.gz", "*.cif.zst") for path in root.rglob(pattern))
        if not self.paths:
            raise ValueError(f"No CIF files found under {data_dir}.")

    def __len__(self) -> int:
        return len(self.paths)

    def _features(self, path: Path) -> dict[str, torch.Tensor]:
        cached = None if self.cache_dir is None else self.cache_dir / f"{path.stem}-{self.seed}.pt"
        if cached is not None and cached.is_file():
            return torch.load(cached, map_location="cpu", weights_only=True)
        features, _ = self._prepare_cif(path, seed=self.seed, device=torch.device("cpu"))
        features = {name: value for name, value in features.items() if isinstance(value, torch.Tensor)}
        if cached is not None:
            torch.save(features, cached)
        return features

    def __getitem__(self, index: int) -> dict[str, torch.Tensor] | None:
        path = self.paths[index]
        try:
            features = self._features(path)
        except ValueError as error:
            # Preparation rejects a structure the model cannot be trained on, such as one with unusable coordinates.
            logger.warning(f"Skipping {path.name}: {error}")
            return None
        # A structure that does not fit is dropped from its batch rather than truncated.
        if features["padding_mask"].shape[1] > self.max_tokens:
            return None
        return features


def collate_fn(examples):
    """Keep one structure per batch: token counts differ per structure, and the model has no padding-free path.

    Raise the effective batch size with `--gradient_accumulation_steps` instead.
    """
    examples = [example for example in examples if example is not None]
    return examples[0] if examples else None


def sample_times(batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
    """Draw one time per modality and batch row from the released training distribution."""
    times = {}
    for name, parameters in TIME_DISTRIBUTION.items():
        beta = torch.distributions.Beta(parameters["concentration1"], parameters["concentration0"])
        sampled = beta.sample((batch_size,)).to(device)
        uniform = torch.rand((batch_size,), device=device)
        take_uniform = torch.rand((batch_size,), device=device) < parameters["uniform_fraction"]
        # `add_noise` rejects a time of exactly 1, where the state is clean and the loss carries no signal.
        times[name] = torch.where(take_uniform, uniform, sampled).clamp(max=1.0 - 1e-6)
    return times


def draw_noise(
    name: str, shape: tuple[int, ...], emit_mask: torch.Tensor, conditioning: dict[str, torch.Tensor]
) -> torch.Tensor:
    """Draw one modality's reference noise the way the sampler draws its starting state.

    Training and sampling have to agree on what the noise end of the path looks like, so the draw reuses the
    pipeline's own helpers: masked to the emitted tokens, and for the backbone with zero centre of mass and the
    per-molecule stochastic offset the released recipe applies.
    """
    noise = RFD4ProteinaPipeline._sample_noise(shape, emit_mask, None, zero_com=name == BACKBONE)
    if name != BACKBONE:
        return noise
    return RFD4ProteinaPipeline._center_noise(noise, emit_mask, None, conditioning)


def flow_matching_loss(prediction, state, times, clean, supervision_masks) -> torch.Tensor:
    """Average the per-modality error of the predicted clean sample over the supervised tokens.

    The model predicts a velocity; `state + (1 - t) * velocity` is its estimate of the clean sample, which is what the
    released recipe scores. Each modality is normalised by its own token count and the two terms are summed.
    """
    velocities = {"bb_ca": prediction.backbone.float(), "local_latents": prediction.local_latents.float()}
    loss = 0.0
    for name, velocity in velocities.items():
        mask = supervision_masks[name]
        estimate = state[name] + (1 - times[name][..., None, None]) * velocity
        error = (clean[name] - estimate) * mask[..., None]
        tokens = mask.sum(dim=-1).clamp(min=1)
        loss = loss + ((error**2).sum(dim=(-1, -2)) / tokens).mean()
    return loss


def main():
    args = parse_args()
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=ProjectConfiguration(project_dir=args.output_dir, logging_dir=str(logging_dir)),
    )
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    weight_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(accelerator.mixed_precision, torch.float32)
    load_kwargs = {"revision": args.revision, "variant": args.variant, "cache_dir": args.cache_dir}
    transformer = RFD4ProteinaTransformer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="transformer", torch_dtype=weight_dtype, **load_kwargs
    )
    vae = RFD4ProteinaAutoencoder.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", torch_dtype=weight_dtype, **load_kwargs
    )
    # One scheduler per modality, as the pipeline holds them; training only uses their shared `add_noise`.
    schedulers = {
        name: RFD4ProteinaScheduler.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder=subfolder,
            cache_dir=args.cache_dir,
            revision=args.revision,
        )
        for name, subfolder in (("bb_ca", "bb_ca_scheduler"), ("local_latents", "local_latents_scheduler"))
    }

    # The autoencoder is paired with the denoiser and stays frozen; only the adapters on the denoiser train.
    vae.requires_grad_(False)
    transformer.requires_grad_(False)
    vae.to(accelerator.device)
    transformer.to(accelerator.device)
    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    transformer.add_adapter(
        LoraConfig(
            r=args.rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            init_lora_weights="gaussian",
            target_modules=args.target_modules,
        )
    )
    if weight_dtype != torch.float32:
        cast_training_params([transformer], dtype=torch.float32)

    def unwrap_model(model):
        return accelerator.unwrap_model(model)

    def save_model_hook(models, weights, output_dir):
        if not accelerator.is_main_process:
            return
        for model in models:
            if isinstance(unwrap_model(model), type(unwrap_model(transformer))):
                layers = convert_state_dict_to_diffusers(get_peft_model_state_dict(unwrap_model(model)))
            else:
                raise ValueError(f"Unexpected model to save: {model.__class__}")
            weights.pop()
        RFD4ProteinaPipeline.save_lora_weights(output_dir, transformer_lora_layers=layers)

    def load_model_hook(models, input_dir):
        transformer_ = None
        while models:
            model = models.pop()
            if isinstance(unwrap_model(model), type(unwrap_model(transformer))):
                transformer_ = unwrap_model(model)
            else:
                raise ValueError(f"Unexpected model to load: {model.__class__}")
        state_dict = RFD4ProteinaPipeline.lora_state_dict(input_dir)
        state_dict = {
            key.removeprefix("transformer."): value
            for key, value in state_dict.items()
            if key.startswith("transformer.")
        }
        incompatible = set_peft_model_state_dict(
            transformer_, convert_unet_state_dict_to_peft(state_dict), adapter_name="default"
        )
        if getattr(incompatible, "unexpected_keys", None):
            logger.warning(f"Unexpected keys in the checkpoint: {incompatible.unexpected_keys}")
        if accelerator.mixed_precision == "fp16":
            cast_training_params([transformer_])

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    parameters = [parameter for parameter in transformer.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    dataset = PreparedStructureDataset(
        args.instance_data_dir, seed=args.prep_seed, max_tokens=args.max_tokens, cache_dir=args.features_cache_dir
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
    )
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )
    transformer, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, dataloader, lr_scheduler
    )
    if accelerator.is_main_process:
        # Trackers take scalars, so the list of adapted modules is logged as one string.
        accelerator.init_trackers(
            "rfd4-proteina-lora", config={**vars(args), "target_modules": ",".join(args.target_modules)}
        )

    logger.info("***** Running training *****")
    logger.info(f"  Structures = {len(dataset)}")
    logger.info(f"  Optimizer steps = {args.max_train_steps}")
    logger.info(f"  Trainable parameters = {sum(parameter.numel() for parameter in parameters)}")

    global_step = 0
    if args.resume_from_checkpoint:
        path = args.resume_from_checkpoint
        if path == "latest":
            checkpoints = sorted(
                (entry for entry in os.listdir(args.output_dir) if entry.startswith("checkpoint")),
                key=lambda entry: int(entry.split("-")[1]),
            )
            path = os.path.join(args.output_dir, checkpoints[-1]) if checkpoints else None
        if path is None:
            logger.info("No checkpoint found; starting a new run.")
        else:
            accelerator.load_state(path)
            global_step = int(Path(path).name.split("-")[1])
            logger.info(f"Resuming from {path} at step {global_step}.")

    progress_bar = tqdm(
        range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process, desc="Steps"
    )
    # Structures the preparation rejects drop out of their batch, so the data is cycled until the step count is met.
    while global_step < args.max_train_steps:
        transformer.train()
        for batch in dataloader:
            if batch is None:  # Every structure in this batch was skipped.
                continue
            with accelerator.accumulate(transformer):
                features = {name: value.to(accelerator.device) for name, value in batch.items()}
                padding_mask = features["padding_mask"].bool()
                conditioning = {name: value for name, value in features.items() if name not in PIPELINE_INPUTS}
                conditioning["is_in_context_conditioning"] = features["is_in_context_conditioning"]
                conditioning["supports_local_latent"] = features["supports_local_latent"]
                emitted = padding_mask & ~features["is_in_context_conditioning"].bool()
                emit_masks = {
                    "bb_ca": emitted,
                    "local_latents": emitted & features["supports_local_latent"].bool(),
                }
                # An experimental structure is missing atoms the residue should have: their slots are marked present
                # but hold no coordinate. Drop them from the mask and zero them, or the encoder returns NaN latents
                # for the whole structure.
                resolved = torch.isfinite(features["coords_nm"]).all(dim=-1)
                atom37_mask = features["atom37_mask"].bool() & resolved
                coords_nm = torch.nan_to_num(features["coords_nm"])
                # Only tokens whose CA atom is resolved carry a target.
                supervision_masks = {
                    "bb_ca": emit_masks["bb_ca"] & atom37_mask[..., 1],
                    "local_latents": emit_masks["local_latents"] & atom37_mask[..., 1],
                }
                if not supervision_masks["bb_ca"].any():
                    continue

                with torch.no_grad():
                    encoded = vae.encode(
                        coords_nm=coords_nm.to(weight_dtype),
                        padding_mask=padding_mask,
                        token_type=features["token_type"],
                        atom37_mask=atom37_mask,
                        conditioning=conditioning,
                    )
                clean = {"bb_ca": coords_nm[..., 1, :].float(), "local_latents": encoded.latent.float()}
                clean = {name: value * emit_masks[name][..., None] for name, value in clean.items()}
                noise = {
                    name: draw_noise(name, tuple(clean[name].shape), emit_masks[name], conditioning) for name in clean
                }
                times = sample_times(padding_mask.shape[0], accelerator.device)
                state = {
                    name: schedulers[name].add_noise(clean[name], noise[name], times[name])
                    * emit_masks[name][..., None]
                    for name in clean
                }

                prediction = transformer(
                    bb_ca=state["bb_ca"].to(weight_dtype),
                    local_latents=state["local_latents"].to(weight_dtype),
                    bb_ca_time=times["bb_ca"],
                    local_latents_time=times["local_latents"],
                    padding_mask=padding_mask,
                    token_type=features["token_type"],
                    atom37_mask=atom37_mask,
                    conditioning=conditioning,
                    recycle_depth=args.recycle_depth,
                )
                loss = flow_matching_loss(prediction, state, times, clean, supervision_masks)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(parameters, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

                if accelerator.is_main_process and global_step % args.checkpointing_steps == 0:
                    save_checkpoint(accelerator, args, global_step)
                if args.validation_cif is not None and global_step % args.validation_steps == 0:
                    run_validation(accelerator, args, unwrap_model(transformer), vae, schedulers, global_step)
                    transformer.train()
            if global_step >= args.max_train_steps:
                break
        if global_step == 0:
            raise ValueError(
                f"No structure under {args.instance_data_dir} could be trained on; see the warnings above."
            )

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        layers = convert_state_dict_to_diffusers(get_peft_model_state_dict(unwrap_model(transformer)))
        RFD4ProteinaPipeline.save_lora_weights(args.output_dir, transformer_lora_layers=layers)
        logger.info(f"Saved the adapter to {args.output_dir}.")
    accelerator.end_training()


def save_checkpoint(accelerator, args, global_step: int) -> None:
    if args.checkpoints_total_limit is not None:
        checkpoints = sorted(
            (entry for entry in os.listdir(args.output_dir) if entry.startswith("checkpoint")),
            key=lambda entry: int(entry.split("-")[1]),
        )
        for checkpoint in checkpoints[: max(len(checkpoints) - args.checkpoints_total_limit + 1, 0)]:
            shutil.rmtree(os.path.join(args.output_dir, checkpoint))
    path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
    accelerator.save_state(path)
    logger.info(f"Saved state to {path}.")


def run_validation(accelerator, args, transformer, vae, schedulers, global_step: int) -> None:
    """Generate one structure from the validation input with the adapter attached."""
    from rfd4_proteina_prep import prepare_cif, write_cif

    pipeline = RFD4ProteinaPipeline(
        transformer=transformer,
        vae=vae,
        bb_ca_scheduler=schedulers["bb_ca"],
        local_latents_scheduler=schedulers["local_latents"],
    )
    pipeline.set_progress_bar_config(disable=True)
    features, atom_array = prepare_cif(args.validation_cif, seed=args.prep_seed, device=accelerator.device)
    inputs = {name: features[name] for name in PIPELINE_INPUTS}
    conditioning = {name: value for name, value in features.items() if name not in PIPELINE_INPUTS}
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed or 0)
    with torch.no_grad():
        output = pipeline(
            **inputs,
            conditioning_features=conditioning,
            generator=generator,
        )
    path = os.path.join(args.output_dir, f"validation-{global_step}.cif")
    write_cif(output, features, atom_array, path)
    logger.info(f"Wrote {path}.")


if __name__ == "__main__":
    main()
