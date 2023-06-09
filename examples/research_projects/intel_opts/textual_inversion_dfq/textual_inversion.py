import argparse
import itertools
import math
import os
import random
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import PIL
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import HfFolder, Repository, whoami
from neural_compressor.utils import logger
from packaging import version
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler


if version.parse(version.parse(PIL.__version__).base_version) >= version.parse("9.1.0"):
    PIL_INTERPOLATION = {
        "linear": PIL.Image.Resampling.BILINEAR,
        "bilinear": PIL.Image.Resampling.BILINEAR,
        "bicubic": PIL.Image.Resampling.BICUBIC,
        "lanczos": PIL.Image.Resampling.LANCZOS,
        "nearest": PIL.Image.Resampling.NEAREST,
    }
else:
    PIL_INTERPOLATION = {
        "linear": PIL.Image.LINEAR,
        "bilinear": PIL.Image.BILINEAR,
        "bicubic": PIL.Image.BICUBIC,
        "lanczos": PIL.Image.LANCZOS,
        "nearest": PIL.Image.NEAREST,
    }
# ------------------------------------------------------------------------------


def save_progress(text_encoder, placeholder_token_id, accelerator, args, save_path):
    logger.info("Saving embeddings")
    learned_embeds = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[placeholder_token_id]
    learned_embeds_dict = {args.placeholder_token: learned_embeds.detach().cpu()}
    torch.save(learned_embeds_dict, save_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Example of distillation for quantization on Textual Inversion.")
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save learned_embeds.bin every X updates steps.",
    )
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
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--train_data_dir", type=str, default=None, required=True, help="A folder containing the training data."
    )
    parser.add_argument(
        "--placeholder_token",
        type=str,
        default=None,
        required=True,
        help="A token to use as a placeholder for the concept.",
    )
    parser.add_argument(
        "--initializer_token", type=str, default=None, required=True, help="A token to use as initializer word."
    )
    parser.add_argument("--learnable_property", type=str, default="object", help="Choose between 'object' and 'style'")
    parser.add_argument("--repeats", type=int, default=100, help="How many times to repeat the training data.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="text-inversion-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
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
        "--center_crop", action="store_true", help="Whether to center crop images before resizing to resolution"
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=5000,
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
        default=1e-4,
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
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
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
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--do_quantization", action="store_true", help="Whether or not to do quantization.")
    parser.add_argument("--do_distillation", action="store_true", help="Whether or not to do distillation.")
    parser.add_argument(
        "--verify_loading", action="store_true", help="Whether or not to verify the loading of the quantized model."
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.train_data_dir is None:
        raise ValueError("You must specify a train data directory.")

    return args


imagenet_templates_small = [
    "a photo of a {}",
    "a rendering of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
    "a photo of a clean {}",
    "a photo of a dirty {}",
    "a dark photo of the {}",
    "a photo of my {}",
    "a photo of the cool {}",
    "a close-up photo of a {}",
    "a bright photo of the {}",
    "a cropped photo of a {}",
    "a photo of the {}",
    "a good photo of the {}",
    "a photo of one {}",
    "a close-up photo of the {}",
    "a rendition of the {}",
    "a photo of the clean {}",
    "a rendition of a {}",
    "a photo of a nice {}",
    "a good photo of a {}",
    "a photo of the nice {}",
    "a photo of the small {}",
    "a photo of the weird {}",
    "a photo of the large {}",
    "a photo of a cool {}",
    "a photo of a small {}",
]

imagenet_style_templates_small = [
    "a painting in the style of {}",
    "a rendering in the style of {}",
    "a cropped painting in the style of {}",
    "the painting in the style of {}",
    "a clean painting in the style of {}",
    "a dirty painting in the style of {}",
    "a dark painting in the style of {}",
    "a picture in the style of {}",
    "a cool painting in the style of {}",
    "a close-up painting in the style of {}",
    "a bright painting in the style of {}",
    "a cropped painting in the style of {}",
    "a good painting in the style of {}",
    "a close-up painting in the style of {}",
    "a rendition in the style of {}",
    "a nice painting in the style of {}",
    "a small painting in the style of {}",
    "a weird painting in the style of {}",
    "a large painting in the style of {}",
]


# Adapted from torch-ema https://github.com/fadel/pytorch_ema/blob/master/torch_ema/ema.py#L14
class EMAModel:
    """
    Exponential Moving Average of models weights
    """

    def __init__(self, parameters: Iterable[torch.nn.Parameter], decay=0.9999):
        parameters = list(parameters)
        self.shadow_params = [p.clone().detach() for p in parameters]

        self.decay = decay
        self.optimization_step = 0

    def get_decay(self, optimization_step):
        """
        Compute the decay factor for the exponential moving average.
        """
        value = (1 + optimization_step) / (10 + optimization_step)
        return 1 - min(self.decay, value)

    @torch.no_grad()
    def step(self, parameters):
        parameters = list(parameters)

        self.optimization_step += 1
        self.decay = self.get_decay(self.optimization_step)

        for s_param, param in zip(self.shadow_params, parameters):
            if param.requires_grad:
                tmp = self.decay * (s_param - param)
                s_param.sub_(tmp)
            else:
                s_param.copy_(param)

        torch.cuda.empty_cache()

    def copy_to(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        """
        Copy current averaged parameters into given collection of parameters.
        Args:
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                updated with the stored moving averages. If `None`, the
                parameters with which this `ExponentialMovingAverage` was
                initialized will be used.
        """
        parameters = list(parameters)
        for s_param, param in zip(self.shadow_params, parameters):
            param.data.copy_(s_param.data)

    def to(self, device=None, dtype=None) -> None:
        r"""Move internal buffers of the ExponentialMovingAverage to `device`.
        Args:
            device: like `device` argument to `torch.Tensor.to`
        """
        # .to() on the tensors handles None correctly
        self.shadow_params = [
            p.to(device=device, dtype=dtype) if p.is_floating_point() else p.to(device=device)
            for p in self.shadow_params
        ]


class TextualInversionDataset(Dataset):
    def __init__(
        self,
        data_root,
        tokenizer,
        learnable_property="object",  # [object, style]
        size=512,
        repeats=100,
        interpolation="bicubic",
        flip_p=0.5,
        set="train",
        placeholder_token="*",
        center_crop=False,
    ):
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.learnable_property = learnable_property
        self.size = size
        self.placeholder_token = placeholder_token
        self.center_crop = center_crop
        self.flip_p = flip_p

        self.image_paths = [os.path.join(self.data_root, file_path) for file_path in os.listdir(self.data_root)]

        self.num_images = len(self.image_paths)
        self._length = self.num_images

        if set == "train":
            self._length = self.num_images * repeats

        self.interpolation = {
            "linear": PIL_INTERPOLATION["linear"],
            "bilinear": PIL_INTERPOLATION["bilinear"],
            "bicubic": PIL_INTERPOLATION["bicubic"],
            "lanczos": PIL_INTERPOLATION["lanczos"],
        }[interpolation]

        self.templates = imagenet_style_templates_small if learnable_property == "style" else imagenet_templates_small
        self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        image = Image.open(self.image_paths[i % self.num_images])

        if not image.mode == "RGB":
            image = image.convert("RGB")

        placeholder_string = self.placeholder_token
        text = random.choice(self.templates).format(placeholder_string)

        example["input_ids"] = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)

        if self.center_crop:
            crop = min(img.shape[0], img.shape[1])
            (
                h,
                w,
            ) = (
                img.shape[0],
                img.shape[1],
            )
            img = img[(h - crop) // 2 : (h + crop) // 2, (w - crop) // 2 : (w + crop) // 2]

        image = Image.fromarray(img)
        image = image.resize((self.size, self.size), resample=self.interpolation)

        image = self.flip_transform(image)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)

        example["pixel_values"] = torch.from_numpy(image).permute(2, 0, 1)
        return example


def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


def freeze_params(params):
    for param in params:
        param.requires_grad = False


def image_grid(imgs, rows, cols):
    if not len(imgs) == rows * cols:
        raise ValueError("The specified number of rows and columns are not correct.")

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def generate_images(pipeline, prompt="", guidance_scale=7.5, num_inference_steps=50, num_images_per_prompt=1, seed=42):
    generator = torch.Generator(pipeline.device).manual_seed(seed)
    images = pipeline(
        prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=generator,
        num_images_per_prompt=num_images_per_prompt,
    ).images
    _rows = int(math.sqrt(num_images_per_prompt))
    grid = image_grid(images, rows=_rows, cols=num_images_per_prompt // _rows)
    return grid


def main():
    args = parse_args()
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        project_config=accelerator_project_config,
    )

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            repo = Repository(args.output_dir, clone_from=repo_name)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load the tokenizer and add the placeholder token as a additional special token
    if args.tokenizer_name:
        tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer_name)
    elif args.pretrained_model_name_or_path:
        tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")

    # Load models and create wrapper for stable diffusion
    noise_scheduler = DDPMScheduler.from_config(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        revision=args.revision,
    )

    train_unet = False
    # Freeze vae and unet
    freeze_params(vae.parameters())
    if not args.do_quantization and not args.do_distillation:
        # Add the placeholder token in tokenizer
        num_added_tokens = tokenizer.add_tokens(args.placeholder_token)
        if num_added_tokens == 0:
            raise ValueError(
                f"The tokenizer already contains the token {args.placeholder_token}. Please pass a different"
                " `placeholder_token` that is not already in the tokenizer."
            )

        # Convert the initializer_token, placeholder_token to ids
        token_ids = tokenizer.encode(args.initializer_token, add_special_tokens=False)
        # Check if initializer_token is a single token or a sequence of tokens
        if len(token_ids) > 1:
            raise ValueError("The initializer token must be a single token.")

        initializer_token_id = token_ids[0]
        placeholder_token_id = tokenizer.convert_tokens_to_ids(args.placeholder_token)
        # Resize the token embeddings as we are adding new special tokens to the tokenizer
        text_encoder.resize_token_embeddings(len(tokenizer))

        # Initialise the newly added placeholder token with the embeddings of the initializer token
        token_embeds = text_encoder.get_input_embeddings().weight.data
        token_embeds[placeholder_token_id] = token_embeds[initializer_token_id]

        freeze_params(unet.parameters())
        # Freeze all parameters except for the token embeddings in text encoder
        params_to_freeze = itertools.chain(
            text_encoder.text_model.encoder.parameters(),
            text_encoder.text_model.final_layer_norm.parameters(),
            text_encoder.text_model.embeddings.position_embedding.parameters(),
        )
        freeze_params(params_to_freeze)
    else:
        train_unet = True
        freeze_params(text_encoder.parameters())

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(
        # only optimize the unet or embeddings of text_encoder
        unet.parameters() if train_unet else text_encoder.get_input_embeddings().parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    train_dataset = TextualInversionDataset(
        data_root=args.train_data_dir,
        tokenizer=tokenizer,
        size=args.resolution,
        placeholder_token=args.placeholder_token,
        repeats=args.repeats,
        learnable_property=args.learnable_property,
        center_crop=args.center_crop,
        set="train",
    )
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    if not train_unet:
        text_encoder = accelerator.prepare(text_encoder)
        unet.to(accelerator.device)
        unet.eval()
    else:
        unet = accelerator.prepare(unet)
        text_encoder.to(accelerator.device)
        text_encoder.eval()
    optimizer, train_dataloader, lr_scheduler = accelerator.prepare(optimizer, train_dataloader, lr_scheduler)

    # Move vae to device
    vae.to(accelerator.device)

    # Keep vae in eval model as we don't train these
    vae.eval()

    compression_manager = None

    def train_func(model):
        if train_unet:
            unet_ = model
            text_encoder_ = text_encoder
        else:
            unet_ = unet
            text_encoder_ = model
        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        if overrode_max_train_steps:
            args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        # Afterwards we recalculate our number of training epochs
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

        # We need to initialize the trackers we use, and also store our configuration.
        # The trackers initializes automatically on the main process.
        if accelerator.is_main_process:
            accelerator.init_trackers("textual_inversion", config=vars(args))

        # Train!
        total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {args.max_train_steps}")
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
        progress_bar.set_description("Steps")
        global_step = 0

        if train_unet and args.use_ema:
            ema_unet = EMAModel(unet_.parameters())

        for epoch in range(args.num_train_epochs):
            model.train()
            train_loss = 0.0
            for step, batch in enumerate(train_dataloader):
                with accelerator.accumulate(model):
                    # Convert images to latent space
                    latents = vae.encode(batch["pixel_values"]).latent_dist.sample().detach()
                    latents = latents * 0.18215

                    # Sample noise that we'll add to the latents
                    noise = torch.randn(latents.shape).to(latents.device)
                    bsz = latents.shape[0]
                    # Sample a random timestep for each image
                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
                    ).long()

                    # Add noise to the latents according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                    # Get the text embedding for conditioning
                    encoder_hidden_states = text_encoder_(batch["input_ids"])[0]

                    # Predict the noise residual
                    model_pred = unet_(noisy_latents, timesteps, encoder_hidden_states).sample

                    loss = F.mse_loss(model_pred, noise, reduction="none").mean([1, 2, 3]).mean()
                    if train_unet and compression_manager:
                        unet_inputs = {
                            "sample": noisy_latents,
                            "timestep": timesteps,
                            "encoder_hidden_states": encoder_hidden_states,
                        }
                        loss = compression_manager.callbacks.on_after_compute_loss(unet_inputs, model_pred, loss)

                    # Gather the losses across all processes for logging (if we use distributed training).
                    avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                    train_loss += avg_loss.item() / args.gradient_accumulation_steps

                    # Backpropagate
                    accelerator.backward(loss)

                    if train_unet:
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(unet_.parameters(), args.max_grad_norm)
                    else:
                        # Zero out the gradients for all token embeddings except the newly added
                        # embeddings for the concept, as we only want to optimize the concept embeddings
                        if accelerator.num_processes > 1:
                            grads = text_encoder_.module.get_input_embeddings().weight.grad
                        else:
                            grads = text_encoder_.get_input_embeddings().weight.grad
                        # Get the index for tokens that we want to zero the grads for
                        index_grads_to_zero = torch.arange(len(tokenizer)) != placeholder_token_id
                        grads.data[index_grads_to_zero, :] = grads.data[index_grads_to_zero, :].fill_(0)

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    if train_unet and args.use_ema:
                        ema_unet.step(unet_.parameters())
                    progress_bar.update(1)
                    global_step += 1
                    accelerator.log({"train_loss": train_loss}, step=global_step)
                    train_loss = 0.0
                    if not train_unet and global_step % args.save_steps == 0:
                        save_path = os.path.join(args.output_dir, f"learned_embeds-steps-{global_step}.bin")
                        save_progress(text_encoder_, placeholder_token_id, accelerator, args, save_path)

                logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

                if global_step >= args.max_train_steps:
                    break
            accelerator.wait_for_everyone()

        if train_unet and args.use_ema:
            ema_unet.copy_to(unet_.parameters())

        if not train_unet:
            return text_encoder_

    if not train_unet:
        text_encoder = train_func(text_encoder)
    else:
        import copy

        model = copy.deepcopy(unet)
        confs = []
        if args.do_quantization:
            from neural_compressor import QuantizationAwareTrainingConfig

            q_conf = QuantizationAwareTrainingConfig()
            confs.append(q_conf)

        if args.do_distillation:
            teacher_model = copy.deepcopy(model)

            def attention_fetcher(x):
                return x.sample

            layer_mappings = [
                [
                    [
                        "conv_in",
                    ]
                ],
                [
                    [
                        "time_embedding",
                    ]
                ],
                [["down_blocks.0.attentions.0", attention_fetcher]],
                [["down_blocks.0.attentions.1", attention_fetcher]],
                [
                    [
                        "down_blocks.0.resnets.0",
                    ]
                ],
                [
                    [
                        "down_blocks.0.resnets.1",
                    ]
                ],
                [
                    [
                        "down_blocks.0.downsamplers.0",
                    ]
                ],
                [["down_blocks.1.attentions.0", attention_fetcher]],
                [["down_blocks.1.attentions.1", attention_fetcher]],
                [
                    [
                        "down_blocks.1.resnets.0",
                    ]
                ],
                [
                    [
                        "down_blocks.1.resnets.1",
                    ]
                ],
                [
                    [
                        "down_blocks.1.downsamplers.0",
                    ]
                ],
                [["down_blocks.2.attentions.0", attention_fetcher]],
                [["down_blocks.2.attentions.1", attention_fetcher]],
                [
                    [
                        "down_blocks.2.resnets.0",
                    ]
                ],
                [
                    [
                        "down_blocks.2.resnets.1",
                    ]
                ],
                [
                    [
                        "down_blocks.2.downsamplers.0",
                    ]
                ],
                [
                    [
                        "down_blocks.3.resnets.0",
                    ]
                ],
                [
                    [
                        "down_blocks.3.resnets.1",
                    ]
                ],
                [
                    [
                        "up_blocks.0.resnets.0",
                    ]
                ],
                [
                    [
                        "up_blocks.0.resnets.1",
                    ]
                ],
                [
                    [
                        "up_blocks.0.resnets.2",
                    ]
                ],
                [
                    [
                        "up_blocks.0.upsamplers.0",
                    ]
                ],
                [["up_blocks.1.attentions.0", attention_fetcher]],
                [["up_blocks.1.attentions.1", attention_fetcher]],
                [["up_blocks.1.attentions.2", attention_fetcher]],
                [
                    [
                        "up_blocks.1.resnets.0",
                    ]
                ],
                [
                    [
                        "up_blocks.1.resnets.1",
                    ]
                ],
                [
                    [
                        "up_blocks.1.resnets.2",
                    ]
                ],
                [
                    [
                        "up_blocks.1.upsamplers.0",
                    ]
                ],
                [["up_blocks.2.attentions.0", attention_fetcher]],
                [["up_blocks.2.attentions.1", attention_fetcher]],
                [["up_blocks.2.attentions.2", attention_fetcher]],
                [
                    [
                        "up_blocks.2.resnets.0",
                    ]
                ],
                [
                    [
                        "up_blocks.2.resnets.1",
                    ]
                ],
                [
                    [
                        "up_blocks.2.resnets.2",
                    ]
                ],
                [
                    [
                        "up_blocks.2.upsamplers.0",
                    ]
                ],
                [["up_blocks.3.attentions.0", attention_fetcher]],
                [["up_blocks.3.attentions.1", attention_fetcher]],
                [["up_blocks.3.attentions.2", attention_fetcher]],
                [
                    [
                        "up_blocks.3.resnets.0",
                    ]
                ],
                [
                    [
                        "up_blocks.3.resnets.1",
                    ]
                ],
                [
                    [
                        "up_blocks.3.resnets.2",
                    ]
                ],
                [["mid_block.attentions.0", attention_fetcher]],
                [
                    [
                        "mid_block.resnets.0",
                    ]
                ],
                [
                    [
                        "mid_block.resnets.1",
                    ]
                ],
                [
                    [
                        "conv_out",
                    ]
                ],
            ]
            layer_names = [layer_mapping[0][0] for layer_mapping in layer_mappings]
            if not set(layer_names).issubset([n[0] for n in model.named_modules()]):
                raise ValueError(
                    "Provided model is not compatible with the default layer_mappings, "
                    'please use the model fine-tuned from "CompVis/stable-diffusion-v1-4", '
                    "or modify the layer_mappings variable to fit your model."
                    f"\nDefault layer_mappings are as such:\n{layer_mappings}"
                )
            from neural_compressor.config import DistillationConfig, IntermediateLayersKnowledgeDistillationLossConfig

            distillation_criterion = IntermediateLayersKnowledgeDistillationLossConfig(
                layer_mappings=layer_mappings,
                loss_types=["MSE"] * len(layer_mappings),
                loss_weights=[1.0 / len(layer_mappings)] * len(layer_mappings),
                add_origin_loss=True,
            )
            d_conf = DistillationConfig(teacher_model=teacher_model, criterion=distillation_criterion)
            confs.append(d_conf)

        from neural_compressor.training import prepare_compression

        compression_manager = prepare_compression(model, confs)
        compression_manager.callbacks.on_train_begin()
        model = compression_manager.model
        train_func(model)
        compression_manager.callbacks.on_train_end()

        # Save the resulting model and its corresponding configuration in the given directory
        model.save(args.output_dir)

        logger.info(f"Optimized model saved to: {args.output_dir}.")

        # change to framework model for further use
        model = model.model

    # Create the pipeline using using the trained modules and save it.
    templates = imagenet_style_templates_small if args.learnable_property == "style" else imagenet_templates_small
    prompt = templates[0].format(args.placeholder_token)
    if accelerator.is_main_process:
        pipeline = StableDiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            text_encoder=accelerator.unwrap_model(text_encoder),
            vae=vae,
            unet=accelerator.unwrap_model(unet),
            tokenizer=tokenizer,
        )
        pipeline.save_pretrained(args.output_dir)
        pipeline = pipeline.to(unet.device)
        baseline_model_images = generate_images(pipeline, prompt=prompt, seed=args.seed)
        baseline_model_images.save(
            os.path.join(args.output_dir, "{}_baseline_model.png".format("_".join(prompt.split())))
        )

        if not train_unet:
            # Also save the newly trained embeddings
            save_path = os.path.join(args.output_dir, "learned_embeds.bin")
            save_progress(text_encoder, placeholder_token_id, accelerator, args, save_path)
        else:
            setattr(pipeline, "unet", accelerator.unwrap_model(model))
            if args.do_quantization:
                pipeline = pipeline.to(torch.device("cpu"))

            optimized_model_images = generate_images(pipeline, prompt=prompt, seed=args.seed)
            optimized_model_images.save(
                os.path.join(args.output_dir, "{}_optimized_model.png".format("_".join(prompt.split())))
            )

        if args.push_to_hub:
            repo.push_to_hub(commit_message="End of training", blocking=False, auto_lfs_prune=True)

    accelerator.end_training()

    if args.do_quantization and args.verify_loading:
        # Load the model obtained after Intel Neural Compressor quantization
        from neural_compressor.utils.pytorch import load

        loaded_model = load(args.output_dir, model=unet)
        loaded_model.eval()

        setattr(pipeline, "unet", loaded_model)
        if args.do_quantization:
            pipeline = pipeline.to(torch.device("cpu"))

        loaded_model_images = generate_images(pipeline, prompt=prompt, seed=args.seed)
        if loaded_model_images != optimized_model_images:
            logger.info("The quantized model was not successfully loaded.")
        else:
            logger.info("The quantized model was successfully loaded.")


if __name__ == "__main__":
    main()
