import functools
import argparse
import os
import random
import time
from pathlib import Path

import datasets
from datasets import concatenate_datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.debug.profiler as xp
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from huggingface_hub import create_repo, upload_folder
from torchvision import transforms
from torchvision.transforms.functional import crop
from transformers import CLIPTextModel, CLIPTextModelWithProjection, AutoTokenizer
from transformers.trainer_pt_utils import get_module_class_from_name
# from viztracer import VizTracer

from torch._dispatch.python import suspend_functionalization
from torch._subclasses.functional_tensor import disable_functional_mode

from torch_xla.distributed.fsdp import checkpoint_module
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from diffusers.training_utils import compute_snr
from diffusers.utils import is_wandb_available
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card


if is_wandb_available():
    pass

print(f"torch_xla version {torch_xla.__version__}")

PROFILE_DIR = os.environ.get("PROFILE_DIR", None)
CACHE_DIR = os.environ.get("CACHE_DIR", None)
if CACHE_DIR:
    xr.initialize_cache(CACHE_DIR, readonly=False)
xr.use_spmd()
DATASET_NAME_MAPPING = {
    "lambdalabs/naruto-blip-captions": ("image", "text"),
}
PORT = 9012


def save_model_card(
    args,
    repo_id: str,
    repo_folder: str = None,
):
    model_description = f"""
# Text-to-image finetuning - {repo_id}

This pipeline was finetuned from **{args.pretrained_model_name_or_path}** on the **{args.dataset_name}** dataset. \n

## Pipeline usage

You can use the pipeline like so:

```python
import torch
import os
import sys
import  numpy as np

import torch_xla.core.xla_model as xm
from time import time
from typing import Tuple
from diffusers import StableDiffusionPipeline

def main(args):
    device = xm.xla_device()
    model_path = <output_dir>
    pipe = StableDiffusionPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16
    )
    pipe.to(device)
    prompt = ["A naruto with green eyes and red legs."]
    image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
    image.save("naruto.png")

if __name__ == '__main__':
    main()
```

## Training info

These are the key hyperparameters used during training:

* Steps: {args.max_train_steps}
* Learning rate: {args.learning_rate}
* Batch size: {args.train_batch_size}
* Image resolution: {args.resolution}
* Mixed-precision: {args.mixed_precision}

"""

    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="creativeml-openrail-m",
        base_model=args.pretrained_model_name_or_path,
        model_description=model_description,
        inference=True,
    )

    tags = ["stable-diffusion", "stable-diffusion-diffusers", "text-to-image", "diffusers", "diffusers-training"]
    model_card = populate_model_card(model_card, tags=tags)

    model_card.save(os.path.join(repo_folder, "README.md"))


def wrap_module(
    mod: torch.nn.Module, transform, prefix: tuple[str, ...] = tuple()
) -> torch.nn.Module:
    """
    Recursively transforms the modules by calling `transform` on them.

    You may use this to apply sharding, checkpointing, optimization barriers, etc.

    Start from the leaf modules and work our way up, to handle cases where one
    module is the child of another. The child modules will be transformed first,
    and then the parent module will be transformed, possibly with transformed
    children.
    """
    new_children = {}
    for name, child in mod.named_children():
        new_children[name] = wrap_module(child, transform, prefix + (name,))
    for name, new_child in new_children.items():
        mod.set_submodule(name, new_child)
    return transform(mod)

def add_checkpoints(model):
    remat_classes = [get_module_class_from_name(model, "BasicTransformerBlock")]
    # import pdb; pdb.set_trace()
    def maybe_checkpoint(mod):
        if isinstance(mod, tuple(remat_classes)):
            return checkpoint_module(mod)
        return mod
    return wrap_module(model, maybe_checkpoint)

class TrainSD:
    def __init__(
        self,
        weight_dtype,
        device,
        noise_scheduler,
        unet,
        optimizer,
        dataloader,
        args,
    ):
        self.weight_dtype = weight_dtype
        self.device = device
        self.noise_scheduler = noise_scheduler
        self.unet = unet
        self.optimizer = optimizer
        self.args = args
        self.mesh = xs.get_global_mesh()
        self.dataloader = iter(dataloader)
        self.global_step = 0

    def run_optimizer(self):
        self.optimizer.step()

    def start_training(self):
        dataloader_exception = False
        measure_start_step = self.args.measure_start_step
        print("meaure_start_step: ", measure_start_step)
        print("max_train_steps: ", self.args.max_train_steps)
        assert measure_start_step < self.args.max_train_steps
        total_time = 0
        last_time = None
        for step in range(0, self.args.max_train_steps):
            print("step: ", step)
            start_time = time.time()
            batch = next(self.dataloader)
            print(f"dataloading time {time.time()-start_time}")
            if step == measure_start_step and PROFILE_DIR is not None:
                xm.wait_device_ops()
                last_time = time.time()
                xp.trace_detached(f"localhost:{PORT}", PROFILE_DIR, duration_ms=args.profile_duration)
            
            with suspend_functionalization(), disable_functional_mode():
                loss = self.step_fn(
                    batch["model_input"],
                    batch["prompt_embeds"],
                    batch["pooled_prompt_embeds"],
                    batch["original_sizes"],
                    batch["crop_top_lefts"])
            self.global_step += 1

            def print_loss_closure(step, loss):
                print(f"Step: {step}, Loss: {loss}")

            if self.args.print_loss:
                xm.add_step_closure(
                    print_loss_closure,
                    args=(
                        self.global_step,
                        loss,
                    ),
                )
        xm.mark_step()
        if not dataloader_exception:
            xm.wait_device_ops()
            if last_time is not None:
                total_time = time.time() - last_time
                print(f"Average step time: {total_time/(self.args.max_train_steps-measure_start_step)}")
        else:
            print("dataloader exception happen, skip result")
            return

    def step_fn(
        self,
        model_input,
        prompt_embeds,
        pooled_prompt_embeds,
        original_sizes,
        crop_top_lefts
    ):
        start_time = time.time()
        with xp.Trace("optimizer_zero_grad"):
            self.optimizer.zero_grad(True)
        with xp.Trace("forward"):
            noise = torch.randn_like(model_input).to(self.device, dtype=self.weight_dtype)
            bsz = model_input.shape[0]
            timesteps = torch.randint(
                0,
                self.noise_scheduler.config.num_train_timesteps,
                (bsz,),
                device=model_input.device,
            )
            timesteps = timesteps.long()
            noisy_latents = self.noise_scheduler.add_noise(model_input, noise, timesteps)
            noisy_latents = noisy_latents.to(self.device, dtype=self.weight_dtype)
            # time ids
            def compute_time_ids(original_size, crops_coords_top_left):
                # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
                target_size = torch.tensor([self.args.resolution, self.args.resolution]).to(self.device)
                add_time_ids = torch.unsqueeze(torch.cat([original_size, crops_coords_top_left, target_size], axis=0), dim=0)
                return add_time_ids

            add_time_ids = torch.cat(
                [compute_time_ids(s, c) for s, c in zip(original_sizes, crop_top_lefts)]
            )
            # Predict the noise residual
            unet_added_conditions = {"time_ids": add_time_ids}
            unet_added_conditions.update({"text_embeds": pooled_prompt_embeds})
            # breakpoint()
            model_pred = self.unet(
                noisy_latents,
                timesteps,
                prompt_embeds,
                added_cond_kwargs=unet_added_conditions,
                return_dict=False,
            )[0]
            if self.args.prediction_type is not None:
                # set prediction_type of scheduler if defined
                self.noise_scheduler.register_to_config(prediction_type=self.args.prediction_type)
            if self.noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif self.noise_scheduler.config.prediction_type == "v_prediction":
                target = self.noise_scheduler.get_velocity(model_input, noise, timesteps)
            elif self.noise_scheduler.config.prediction_type == "sample":
                # We set the target to latents here, but the model_pred will return the noise sample prediction.
                target = model_input
                # We will have to subtract the noise residual from the prediction to get the target sample.
                model_pred = model_pred - noise
            else:
                raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

        print(f"forward_time = {time.time()-start_time}")
        start_time = time.time()
        with xp.Trace("backward"):
            if self.args.snr_gamma is None:
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            else:
                # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                # This is discussed in Section 4.2 of the same paper.
                snr = compute_snr(self.noise_scheduler, timesteps)
                mse_loss_weights = torch.stack([snr, self.args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(
                    dim=1
                )[0]
                if self.noise_scheduler.config.prediction_type == "epsilon":
                    mse_loss_weights = mse_loss_weights / snr
                elif self.noise_scheduler.config.prediction_type == "v_prediction":
                    mse_loss_weights = mse_loss_weights / (snr + 1)

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                loss = loss.mean()
            loss.backward()
        print(f"backward time = {time.time()-start_time}")
        start_time = time.time()
        with xp.Trace("optimizer_step"):
            self.run_optimizer()
        print(f"optimizer step = {time.time()-start_time}")
        return model_pred


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--profile_duration", type=int, default=10000, help="Profile duration in ms")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing an image."
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--xla_gradient_checkpointing",
        default=False,
        action="store_true",
        help=("Enable gradient checkpointing to save memory at the expense of slower backward pass."
              "This saves the inputs to the BasicTransformerBlock and recomputes the forward pass during the backward pass."
        )
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--loader_prefetch_size",
        type=int,
        default=1,
        help=("Number of subprocesses to use for data loading to cpu."),
    )
    parser.add_argument(
        "--loader_prefetch_factor",
        type=int,
        default=2,
        help=("Number of batches loaded in advance by each worker."),
    )
    parser.add_argument(
        "--device_prefetch_size",
        type=int,
        default=1,
        help=("Number of subprocesses to use for data loading to tpu from cpu. "),
    )
    parser.add_argument("--measure_start_step", type=int, default=10, help="Step to start profiling.")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediction_type` is chosen.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "bf16"],
        help=("Whether to use mixed precision. Bf16 requires PyTorch >= 1.10"),
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--print_loss",
        default=False,
        action="store_true",
        help=("Print loss at every step."),
    )

    args = parser.parse_args()

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args


def setup_optimizer(unet, args):
    optimizer_cls = torch.optim.AdamW
    return optimizer_cls(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
        foreach=True,
    )

def encode_prompt(batch, text_encoders, tokenizers, proportion_empty_prompts, caption_column, dtype, is_train=True):
    prompt_embeds_list = []
    prompt_batch = batch[caption_column]

    captions = []
    for caption in prompt_batch:
        if random.random() < proportion_empty_prompts:
            captions.append("")
        elif isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])

    with torch.no_grad():
        for tokenizer, text_encoder in zip(tokenizers, text_encoders):
            text_inputs = tokenizer(
                captions,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            prompt_embeds = text_encoder(
                text_input_ids.to(text_encoder.device),
                output_hidden_states=True,
                return_dict=False,
            )

            # We are only ALWAYS interested in the pooled output of the final text encoder
            pooled_prompt_embeds = prompt_embeds[0]
            prompt_embeds = prompt_embeds[-1][-2]
            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
            prompt_embeds_list.append(prompt_embeds)

    
    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1).to(dtype=dtype)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1).to(dtype=dtype)
    return {"prompt_embeds": prompt_embeds, "pooled_prompt_embeds": pooled_prompt_embeds}

def compute_vae_encodings(batch, vae):
    images = batch.pop("pixel_values")
    pixel_values = torch.stack(list(images))
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    pixel_values = pixel_values.to(vae.device, dtype=vae.dtype)

    with torch.no_grad():
        model_input = vae.encode(pixel_values).latent_dist.sample()
    model_input = model_input * vae.config.scaling_factor
    xm.mark_step()
    return {"model_input": model_input.cpu()}


def load_dataset(args):
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = datasets.load_dataset(
            args.dataset_name,
            cache_dir=args.cache_dir,
            data_dir=args.train_data_dir,
        )
    else:
        data_files = {}
        if args.train_data_dir is not None:
            data_files["train"] = os.path.join(args.train_data_dir, "**")
        dataset = datasets.load_dataset(
            "imagefolder",
            data_files=data_files,
            cache_dir=args.cache_dir,
        )
    return dataset


def get_column_names(dataset, args):
    column_names = dataset["train"].column_names

    dataset_columns = DATASET_NAME_MAPPING.get(args.dataset_name, None)
    if args.image_column is None:
        image_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        image_column = args.image_column
        if image_column not in column_names:
            raise ValueError(
                f"--image_column' value '{args.image_column}' needs to be one of: {', '.join(column_names)}"
            )
    if args.caption_column is None:
        caption_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        caption_column = args.caption_column
        if caption_column not in column_names:
            raise ValueError(
                f"--caption_column' value '{args.caption_column}' needs to be one of: {', '.join(column_names)}"
            )
    return image_column, caption_column


def main(args):
    args = parse_args()
    cache_path = Path(os.environ.get('CACHE_DIR', "/tmp/xla_cache/"))
    cache_path.mkdir(parents=True, exist_ok=True)
    xr.initialize_cache(str(cache_path), readonly=False)

    server = xp.start_server(PORT)

    num_devices = xr.global_runtime_device_count()
    mesh = xs.get_1d_mesh("data")
    xs.set_global_mesh(mesh)

    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
        variant=args.variant,
    )
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
      args.pretrained_model_name_or_path,
      subfolder="text_encoder_2",
      revision=args.revision,
      variant=args.variant,
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
    )

    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        revision=args.non_ema_revision,
    )

    if xm.is_master_ordinal() and args.push_to_hub:
        repo_id = create_repo(
            repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
        ).repo_id

    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
        use_fast=False
    )
    tokenizer_2 = AutoTokenizer.from_pretrained(
      args.pretrained_model_name_or_path,
      subfolder="tokenizer_2",
      revision=args.revision,
      use_fast=False
    )
    unet.enable_xla_attention(partition_spec=("data", None, None, None))

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    unet.train()

    # For mixed precision training we cast all non-trainable weights (vae,
    # non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full
    # precision is not required.
    weight_dtype = torch.float32
    if args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    device = torch_xla.device()

    # Move text_encode and vae to device and cast to weight_dtype
    text_encoder = text_encoder.to(device, dtype=weight_dtype)
    text_encoder_2 = text_encoder_2.to(device, dtype=weight_dtype)
    vae = vae.to(device, dtype=weight_dtype)
    unet = unet.to(device, dtype=weight_dtype)
    optimizer = setup_optimizer(unet, args)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    unet.train()

    dataset = load_dataset(args)
    image_column, caption_column = get_column_names(dataset, args)

    train_resize = transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR)
    train_crop = transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution)
    train_flip = transforms.RandomHorizontalFlip(p=1.0)
    train_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        original_sizes = []
        all_images = []
        crop_top_lefts = []
        for image in images:
          original_sizes.append((image.height, image.width))
          image = train_resize(image)
          if args.random_flip and random.random() < 0.5:
              # flip
              image = train_flip(image)
          if args.center_crop:
              y1 = max(0, int(round((image.height - args.resolution) / 2.0)))
              x1 = max(0, int(round((image.width - args.resolution) / 2.0)))
              image = train_crop(image)
          else:
              y1, x1, h, w = train_crop.get_params(image, (args.resolution, args.resolution))
              image = crop(image, y1, x1, h, w)
          crop_top_left = (y1, x1)
          crop_top_lefts.append(crop_top_left)
          image = train_transforms(image)
          all_images.append(image)

        examples["original_sizes"] = original_sizes
        examples["crop_top_lefts"] = crop_top_lefts
        examples["pixel_values"] = all_images
        return examples

    train_dataset = dataset["train"]
    train_dataset.set_format("torch")
    train_dataset.set_transform(preprocess_train)

    text_encoders = [text_encoder, text_encoder_2]
    tokenizers = [tokenizer, tokenizer_2]
    compute_embeddings_fn = functools.partial(
      encode_prompt,
      text_encoders=text_encoders,
      tokenizers=tokenizers,
      proportion_empty_prompts=0,
      caption_column=caption_column,
      dtype=weight_dtype
    )
    compute_vae_encodings_fn = functools.partial(compute_vae_encodings, vae=vae)
    from datasets.fingerprint import Hasher
    data_args = (args.pretrained_model_name_or_path, args.dataset_name, args.caption_column, args.image_column, args.resolution, args.center_crop, args.random_flip, args.mixed_precision, args.revision, args.variant)
    new_fingerprint = Hasher.hash(data_args)
    new_fingerprint_for_vae = Hasher.hash((data_args, "vae"))
    train_dataset_with_embeddings = train_dataset.map(
        compute_embeddings_fn, batched=True, batch_size=50, new_fingerprint=new_fingerprint
    )
    train_dataset_with_vae = train_dataset.map(
        compute_vae_encodings_fn,
        batched=True,
        batch_size=50,
        new_fingerprint=new_fingerprint_for_vae,
    )
    precomputed_dataset = concatenate_datasets(
        [train_dataset_with_embeddings, train_dataset_with_vae.remove_columns(["image", "text"])], axis=1
    )
    precomputed_dataset = precomputed_dataset.with_transform(preprocess_train)
    del compute_vae_encodings_fn, compute_embeddings_fn, text_encoder, text_encoder_2
    del text_encoders, tokenizers, vae
    def collate_fn(examples):
        model_input = torch.stack([torch.tensor(example["model_input"]) for example in examples]).to(dtype=weight_dtype)
        original_sizes = torch.stack([torch.tensor(example["original_sizes"]) for example in examples])
        crop_top_lefts = torch.stack([torch.tensor(example["crop_top_lefts"]) for example in examples])
        prompt_embeds = torch.stack([torch.tensor(example["prompt_embeds"]) for example in examples]).to(dtype=weight_dtype)
        pooled_prompt_embeds = torch.stack([torch.tensor(example["pooled_prompt_embeds"]) for example in examples]).to(dtype=weight_dtype)
        return {
          "model_input": model_input,
          "prompt_embeds": prompt_embeds,
          "pooled_prompt_embeds": pooled_prompt_embeds,
          "original_sizes": original_sizes,
          "crop_top_lefts": crop_top_lefts,
        }

    g = torch.Generator()
    g.manual_seed(xr.host_index())
    sampler = torch.utils.data.RandomSampler(precomputed_dataset, replacement=True, num_samples=int(1e10), generator=g)
    train_dataloader = torch.utils.data.DataLoader(
        precomputed_dataset,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
        batch_size=args.train_batch_size,
        prefetch_factor=args.loader_prefetch_factor,
    )

    train_dataloader = pl.MpDeviceLoader(
        train_dataloader,
        device,
        input_sharding={
            "model_input": xs.ShardingSpec(mesh, ("data", None, None, None), minibatch=True),
            "prompt_embeds" : xs.ShardingSpec(mesh, ("data", None, None), minibatch=True),
            "pooled_prompt_embeds" : xs.ShardingSpec(mesh, ("data", None,), minibatch=True),
            "original_sizes" : xs.ShardingSpec(mesh, ("data",), minibatch=True),
            "crop_top_lefts": xs.ShardingSpec(mesh, ("data",), minibatch=True),
        },
        loader_prefetch_size=args.loader_prefetch_size,
        device_prefetch_size=args.device_prefetch_size,
    )

    num_hosts = xr.process_count()
    num_devices_per_host = num_devices // num_hosts
    if xm.is_master_ordinal():
        print("***** Running training *****")
        print(f"Instantaneous batch size per device = {args.train_batch_size // num_devices_per_host }")
        print(
            f"Total train batch size (w. parallel, distributed & accumulation) = {args.train_batch_size * num_hosts}"
        )
        print(f"  Total optimization steps = {args.max_train_steps}")

    if args.xla_gradient_checkpointing:
        unet = add_checkpoints(unet)

    trainer = TrainSD(
        weight_dtype=weight_dtype,
        device=device,
        noise_scheduler=noise_scheduler,
        unet=unet,
        optimizer=optimizer,
        dataloader=train_dataloader,
        args=args,
    )
    trainer.start_training()
    unet = trainer.unet.to("cpu")

    pipeline = StableDiffusionXLPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        unet=unet,
        revision=args.revision,
        variant=args.variant,
    )
    pipeline.save_pretrained(args.output_dir)

    if xm.is_master_ordinal() and args.push_to_hub:
        save_model_card(args, repo_id, repo_folder=args.output_dir)
        upload_folder(
            repo_id=repo_id,
            folder_path=args.output_dir,
            commit_message="End of training",
            ignore_patterns=["step_*", "epoch_*"],
        )


if __name__ == "__main__":
    args = parse_args()
    main(args)