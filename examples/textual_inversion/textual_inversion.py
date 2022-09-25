import argparse
import itertools
import math
import os
import random
from pathlib import Path
from typing import Optional
import sys
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset

import PIL
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, PNDMScheduler, StableDiffusionPipelineMixedDevices, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from huggingface_hub import HfFolder, Repository, whoami
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
import albumentations as A
import gc
import wandb
from torchvision.transforms.functional import InterpolationMode
sys.path.append('./BLIP')
from models.blip import blip_decoder

# logger = get_logger(__name__)
device = "cuda" if torch.cuda.is_available() else "cpu"

def wandb_setup(
    args: dict,
    project_name: str = "glide-text2im-finetune",
):
    return wandb.init(
        project=project_name,
        config=args,
    )

def save_progress(text_encoder, pipeline, placeholder_token_ids, accelerator, args, placeholder_token_concat):
    print("Saving pipeline")
    pipeline.save_pretrained(args.output_dir)
    learned_embeds = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[placeholder_token_ids]
    learned_embeds = text_encoder.get_input_embeddings().weight[placeholder_token_ids]
    learned_embeds_dict = {}

    for placeholder_token in placeholder_token_concat.split(' '):
        learned_embeds_dict[placeholder_token] = learned_embeds[0].detach().cpu()
    torch.save(learned_embeds_dict, os.path.join(args.output_dir, "learned_embeds.bin"))
def get_pipeline(text_encoder, vae, unet, tokenizer,accelerator):
    pipeline = StableDiffusionPipelineMixedDevices(
        text_encoder=accelerator.unwrap_model(text_encoder),
        vae=vae,
        unet=unet,
        tokenizer=tokenizer,
        scheduler=PNDMScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", skip_prk_steps=True
        ),
        feature_extractor=CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32"),
    )
    return pipeline
def log_progress(pipeline, args, step, wandb_run, placeholder_token, logs={}):
    print("Running pipeline")

    prompt = f"A picture of {placeholder_token}"

    with torch.autocast("cuda"):
        image = pipeline(prompt, height=args.resolution, width=args.resolution, num_inference_steps=50, guidance_scale=7.5).images[0]

    image.save("output.png")
    wandb_run.log(
        {
            **logs,
            "iter": step,
            "samples": wandb.Image("output.png", caption=prompt),
        }
    )
def generate_caption(pil_image, blip_model, max_length=100, min_length=50, blip_image_eval_size=384):
    # Idea from https://colab.research.google.com/github/pharmapsychotic/clip-interrogator/blob/main/clip_interrogator.ipynb#scrollTo=30xPxDSDrJEl
    gpu_image = transforms.Compose([
        transforms.Resize((blip_image_eval_size, blip_image_eval_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        caption = blip_model.generate(gpu_image, sample=False, num_beams=3, max_length=max_length, min_length=min_length)
    return caption[0]

def find_longest_common_substring(captions):
    """
    Maybe not optimal but it works. Taken from https://www.geeksforgeeks.org/longest-common-substring-array-strings/
    """
    first_str_len = len(captions[0])
    output = ""
    for i in range(first_str_len):
        for j in range(i+1, first_str_len+1):
            # str1[i], str2[j] is the start of the candidate match
            match = captions[0][i:j]
            valid_substring = True
            for k in range(1, len(captions)):
                if match not in captions[k]:
                    valid_substring=False
                    break
            if valid_substring:
                if len(match) > len(output):
                    output = match

    return output


def predict_replacement_words(images, max_length=100, min_length=50, blip_image_eval_size=384):
    blip_model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_caption_capfilt_large.pth'
    blip_model = blip_decoder(pretrained=blip_model_url, image_size=blip_image_eval_size, med_config='BLIP/configs/med_config.json', vit='base')
    blip_model.eval()
    blip_model = blip_model.to(device)
    captions = []
    for image in images:
        captions.append(generate_caption(Image.open(image), blip_model, max_length, min_length, blip_image_eval_size))
    longest_common_substring = find_longest_common_substring(captions)
    del blip_model
    return longest_common_substring.strip()
    
def add_tokens_and_get_placeholder_token(args, token_ids, tokenizer, text_encoder):
    assert args.num_vec_per_token % len(token_ids) == 0
    placeholder_tokens = [f"{args.placeholder_token}_{i}" for i in range(args.num_vec_per_token)]

    for placeholder_token in placeholder_tokens:
        num_added_tokens = tokenizer.add_tokens(placeholder_token)
        if num_added_tokens == 0:
            raise ValueError(
                f"The tokenizer already contains the token {placeholder_token}. Please pass a different"
                " `placeholder_token` that is not already in the tokenizer."
            )
    placeholder_token = " ".join(placeholder_tokens)
    placeholder_token_ids = tokenizer.encode(placeholder_token, add_special_tokens=False)
    print(f"The placeholder tokens are {placeholder_token} while the ids are {placeholder_token_ids}")
    text_encoder.resize_token_embeddings(len(tokenizer))
    token_embeds = text_encoder.get_input_embeddings().weight.data
    if args.initialize_rest_random:
        # The idea is that the placeholder tokens form adjectives as in x x x white dog.
        for i, placeholder_token_id in enumerate(placeholder_token_ids):
            if len(placeholder_token_ids)-i <len(token_ids):
                token_embeds[placeholder_token_id] = token_embeds[token_ids[i % len(token_ids)]]
            else:
                token_embeds[placeholder_token_id] = torch.rand_like(token_embeds[placeholder_token_id])
    else:
        for i, placeholder_token_id in enumerate(placeholder_token_ids):
            token_embeds[placeholder_token_id] = token_embeds[token_ids[i % len(token_ids)]]
    return placeholder_token, placeholder_token_ids
def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--num_vec_per_token",
        type=int,
        default=1,
        help="The number of vectors used to represent the placeholder token. The higher the number, the better the result at the cost of editability. This can be fixed by prompt editing.",
    )
    parser.add_argument(
        "--guess_initializer_token",
        action="store_true",
        help="Guess the string the represent the concept using blip.",
    )
    parser.add_argument(
        "--initialize_rest_random",
        action="store_true",
        help="Initialize rest of the placeholder tokens with random.",
    )
    parser.add_argument(
        "--slice_div",
        type=int,
        default=1,
        help="The slice amount for less memory usage. Higher slice means less memory but longer computation.",
    )
    parser.add_argument(
        "--project_name",
        type=str,
        default='huggingface_textual_inv',
        help="Name of wandb run",
    )
    parser.add_argument(
        "--log_frequency",
        type=int,
        default=100,
        help="Frequency to log/save the model.",
    )
    parser.add_argument(
        "--save_frequency",
        type=int,
        default=500,
        help="Frequency to log/save the model.",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
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
        "--center_crop", action="store_true", help="Whether to center crop images before resizing to resolution"
    )
    parser.add_argument(
        "--random_crop", action="store_true", help="Whether to random crop images before resizing to resolution"
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
        default=True,
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
    parser.add_argument(
        "--use_auth_token",
        action="store_true",
        help=(
            "Will use the token generated when running `huggingface-cli login` (necessary to use this script with"
            " private models)."
        ),
    )
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
        random_crop=False
    ):
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.learnable_property = learnable_property
        self.size = size
        self.placeholder_token = placeholder_token
        self.center_crop = center_crop
        self.random_crop = random_crop
        self.flip_p = flip_p

        self.image_paths = [os.path.join(self.data_root, file_path) for file_path in os.listdir(self.data_root)]

        self.num_images = len(self.image_paths)
        self._length = self.num_images

        if set == "train":
            self._length = self.num_images * repeats

        self.interpolation = {
            "linear": PIL.Image.LINEAR,
            "bilinear": PIL.Image.BILINEAR,
            "bicubic": PIL.Image.BICUBIC,
            "lanczos": PIL.Image.LANCZOS,
        }[interpolation]
        resize_ratio=0.75
        self.base_transform = A.Compose([
                A.RandomResizedCrop(self.size, self.size, scale=(resize_ratio, 1), ratio=(1, 1), p=1)
            ],
        )

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
        if self.random_crop:
            transformed = self.base_transform(image=img)
            img = transformed["image"]

        if self.center_crop:
            crop = min(img.shape[0], img.shape[1])
            h, w, = (
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


def main():
    args = parse_args()
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    wandb_run = wandb_setup(args, args.project_name)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        logging_dir=logging_dir,
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
        tokenizer = CLIPTokenizer.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="tokenizer", use_auth_token=args.use_auth_token
        )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", use_auth_token=args.use_auth_token
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", use_auth_token=args.use_auth_token
    )
    # Add the placeholder token in tokenizer
    # Idea: add tokens as f"{args.placeholder_token}_i" and just have the combination of all that be the placeholder token
    if args.guess_initializer_token:
        # 
        guessed_concept = predict_replacement_words([os.path.join(args.train_data_dir, file_path) for file_path in os.listdir(args.train_data_dir)])
        token_ids = tokenizer.encode(guessed_concept, add_special_tokens=False)
        print(f"Guessed concept is {guessed_concept} token ids are {token_ids}")
        placeholder_token, placeholder_token_ids = add_tokens_and_get_placeholder_token(args, token_ids, tokenizer, text_encoder)
    else:
        token_ids = tokenizer.encode(args.initializer_token, add_special_tokens=False)
        # regardless of whether the number of token_ids is 1 or more, it'll set one and then keep repeating.
        placeholder_token, placeholder_token_ids = add_tokens_and_get_placeholder_token(args, token_ids, tokenizer, text_encoder)
    # Load models and create wrapper for stable diffusion
    
    
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", use_auth_token=args.use_auth_token
    )
    slice_size = unet.config.attention_head_dim // args.slice_div
    unet.set_attention_slice(slice_size)

    # Resize the token embeddings as we are adding new special tokens to the tokenizer

    # Initialise the newly added placeholder token with the embeddings of the initializer token
    # Freeze vae and unet
    freeze_params(vae.parameters())
    freeze_params(unet.parameters())
    # Freeze all parameters except for the token embeddings in text encoder
    params_to_freeze = itertools.chain(
        text_encoder.text_model.encoder.parameters(),
        text_encoder.text_model.final_layer_norm.parameters(),
        text_encoder.text_model.embeddings.position_embedding.parameters(),
    )
    freeze_params(params_to_freeze)

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(
        text_encoder.get_input_embeddings().parameters(),  # only optimize the embeddings
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # TODO (patil-suraj): laod scheduler using args
    noise_scheduler = DDPMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000, tensor_format="pt"
    )

    train_dataset = TextualInversionDataset(
        data_root=args.train_data_dir,
        tokenizer=tokenizer,
        size=args.resolution,
        placeholder_token=placeholder_token,
        repeats=args.repeats,
        learnable_property=args.learnable_property,
        center_crop=args.center_crop,
        random_crop=args.random_crop,
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
    text_encoder, optimizer, lr_scheduler = accelerator.prepare(
        text_encoder, optimizer, lr_scheduler
    )
    torch.cuda.empty_cache()
    # Move vae and unet to device
    vae.to('cpu')   
    unet.to(device)
    # Keep vae and unet in eval model as we don't train these
    vae.eval()
    unet.eval()
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
    # total_batch_size = args.train_batch_size * args.gradient_accumulation_steps


    print("***** Running training *****")
    print(f"  Num examples = {len(train_dataset)}")
    print(f"  Num Epochs = {args.num_train_epochs}")
    print(f"  Instantaneous batch size per device = {args.train_batch_size}")
    print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    print(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps))
    progress_bar.set_description("Steps")
    global_step = 0
    
    for epoch in range(args.num_train_epochs):
        text_encoder.train()
        for step, batch in enumerate(train_dataloader):
            gc.collect()
            torch.cuda.empty_cache()
            with accelerator.accumulate(text_encoder):

                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"]).latent_dist.sample().detach()
                latents = latents * 0.18215

                # Sample noise that we'll add to the latents
                noise = torch.randn(latents.shape).to(latents.device)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=device).long()
                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps).to(device)
                encoder_hidden_states = text_encoder(batch["input_ids"])[0].to(device)
                # Predict the noise residual
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample.to('cpu')
                loss = F.mse_loss(noise_pred, noise, reduction="none").mean([1, 2, 3]).mean()

                accelerator.backward(loss.to(device))
                # Zero out the gradients for all token embeddings except the newly added
                # embeddings for the concept, as we only want to optimize the concept embeddings
                if accelerator.num_processes > 1:
                    grads = text_encoder.module.get_input_embeddings().weight.grad
                else:
                    grads = text_encoder.get_input_embeddings().weight.grad
                # Get the index for tokens that we want to zero the grads for
                grad_mask = torch.arange(len(tokenizer)) != placeholder_token_ids[0]
                for i in range(1, len(placeholder_token_ids)):
                    grad_mask = grad_mask & (torch.arange(len(tokenizer)) != placeholder_token_ids[i])
                grads.data[grad_mask, :] = grads.data[grad_mask, :].fill_(0)
                # Adding back weight decay
                with torch.no_grad():
                    text_encoder.get_input_embeddings().weight[~grad_mask, :] -= lr_scheduler.get_last_lr()[0]*args.adam_weight_decay*text_encoder.get_input_embeddings().weight[~grad_mask, :]
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                del noise
                del noisy_latents
                del timesteps
                del encoder_hidden_states
                del noise_pred
                
                del grads
                del grad_mask
                del text_encoder.get_input_embeddings().weight.grad
                gc.collect()
                torch.cuda.empty_cache()
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                if global_step % args.log_frequency == 0:
                    pipeline=get_pipeline(text_encoder, vae, unet, tokenizer, accelerator)
                    log_progress(pipeline, args, global_step, wandb_run, placeholder_token)

                    if global_step % args.save_frequency == 0:
                        save_progress(text_encoder, pipeline, placeholder_token_ids, accelerator, args, placeholder_token)

                    del pipeline

            

            if global_step >= args.max_train_steps:
                break
            logs = {"loss": loss.detach().cpu().item()*args.gradient_accumulation_steps, "lr": lr_scheduler.get_last_lr()[0]}
            wandb_run.log(logs)
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            del loss
            if global_step >= args.max_train_steps:
                break

        accelerator.wait_for_everyone()

    # Create the pipeline using using the trained modules and save it.
    if accelerator.is_main_process:
        pipeline=get_pipeline(text_encoder, vae, unet, tokenizer, accelerator)
        log_progress(pipeline, args, global_step, wandb_run, placeholder_token)
        save_progress(text_encoder, pipeline, placeholder_token_ids, accelerator, args, placeholder_token)
        # Also save the newly trained embeddings

        if args.push_to_hub:
            repo.push_to_hub(
                args, pipeline, repo, commit_message="End of training", blocking=False, auto_lfs_prune=True
            )

    accelerator.end_training()


if __name__ == "__main__":
    main()
