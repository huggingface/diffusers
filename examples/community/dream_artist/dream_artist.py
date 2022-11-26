import argparse
import gc
import itertools
import math
import os
import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset

import albumentations as A
import PIL
import wandb
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    PNDMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker, safety_checker
from huggingface_hub import HfFolder, Repository, whoami
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from tqdm.auto import tqdm
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer


sys.path.append("./BLIP")
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


def save_progress(text_encoder, placeholder_token_ids, accelerator, args, placeholder_token_concat, global_step):
    print("Saving learn embedding")
    learned_embeds = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[placeholder_token_ids]
    learned_embeds_dict = {}
    placeholder_tokens = placeholder_token_concat
    if args.subject_noun:
        placeholder_tokens = placeholder_tokens[:-len(args.subject_noun)-1]
    if args.separate_token_format:
        for i, placeholder_token in enumerate(placeholder_tokens.split(" ")):
            learned_embeds_dict[placeholder_token] = learned_embeds[i].detach().cpu()
    else:
        learned_embeds_dict[args.placeholder_token] = learned_embeds.detach().cpu()
    torch.save(learned_embeds_dict, os.path.join(args.output_dir, f"learned_embeds_{global_step}.bin"))


def get_pipeline(text_encoder, vae, unet, tokenizer, accelerator, weight_dtype):
    # I disabled safety checker as it causes an oom
    pipeline = StableDiffusionPipeline(
        text_encoder=accelerator.unwrap_model(text_encoder),
        vae=vae,
        unet=unet,
        tokenizer=tokenizer,
        scheduler=PNDMScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", skip_prk_steps=True
        ),
        safety_checker=None,
        feature_extractor=None
    )
    return pipeline


def log_progress(pipeline, args, step, wandb_run, placeholder_token, negative_prompt, logs={}):
    print("Running pipeline")

    prompt = f"a photo of a {placeholder_token}"

    with torch.autocast("cuda"):
        image = pipeline(
            prompt, height=args.resolution, width=args.resolution, num_inference_steps=50, guidance_scale=args.guidance_scale, negative_prompt=negative_prompt
        ).images[0]

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
    gpu_image = (
        transforms.Compose(
            [
                transforms.Resize(
                    (blip_image_eval_size, blip_image_eval_size), interpolation=InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ]
        )(pil_image)
        .unsqueeze(0)
        .to(device)
    )

    with torch.no_grad():
        caption = blip_model.generate(
            gpu_image, sample=False, num_beams=3, max_length=max_length, min_length=min_length
        )
    return caption[0]


def find_longest_common_substring(captions):
    """
    Maybe not optimal but it works. Taken from https://www.geeksforgeeks.org/longest-common-substring-array-strings/
    """
    first_str_len = len(captions[0])
    output = ""
    for i in range(first_str_len):
        for j in range(i + 1, first_str_len + 1):
            # str1[i], str2[j] is the start of the candidate match
            match = captions[0][i:j]
            valid_substring = True
            for k in range(1, len(captions)):
                if match not in captions[k]:
                    valid_substring = False
                    break
            if valid_substring:
                if len(match) > len(output):
                    output = match

    return output


def predict_replacement_words(images, max_length=100, min_length=50, blip_image_eval_size=384):
    blip_model_url = (
        "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_caption_capfilt_large.pth"
    )
    blip_model = blip_decoder(
        pretrained=blip_model_url,
        image_size=blip_image_eval_size,
        med_config="BLIP/configs/med_config.json",
        vit="base",
    )
    blip_model.eval()
    blip_model = blip_model.to(device)
    captions = []
    for image in images:
        captions.append(generate_caption(Image.open(image), blip_model, max_length, min_length, blip_image_eval_size))
    longest_common_substring = find_longest_common_substring(captions)
    del blip_model
    return longest_common_substring.strip()


def add_tokens_and_get_placeholder_token(args, token_ids, tokenizer, text_encoder, num_vec_per_token, original_placeholder_token, is_random=False):
    assert num_vec_per_token >= len(token_ids)
    placeholder_tokens = [f"{original_placeholder_token}_{i}" for i in range(num_vec_per_token)]

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
    if is_random or args.subject_noun:
        # Initialize them to be random
        for i, placeholder_token_id in enumerate(placeholder_token_ids):
            token_embeds[placeholder_token_id] = torch.randn_like(token_embeds[placeholder_token_id])
    elif args.initialize_rest_random:
        # The idea is that the placeholder tokens form adjectives as in x x x white dog.
        for i, placeholder_token_id in enumerate(placeholder_token_ids):
            if len(placeholder_token_ids) - i < len(token_ids):
                token_embeds[placeholder_token_id] = token_embeds[token_ids[i % len(token_ids)]]
            else:
                token_embeds[placeholder_token_id] = torch.randn_like(token_embeds[placeholder_token_id])
    else:
        for i, placeholder_token_id in enumerate(placeholder_token_ids):
            token_embeds[placeholder_token_id] = token_embeds[token_ids[i % len(token_ids)]]
    return placeholder_token, placeholder_token_ids

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--use_disc",
        action="store_true",
        help="Use discriminator for training.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Guidance scale for inference.",
    )
    parser.add_argument(
        "--separate_token_format",
        action="store_true",
        help="Saving learned embed in custom format.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--subject_noun",
        type=str,
        default=None,
        help=(
            "Inspired by dream booth. Make the model guess the identifier in the form [identifier] subject noun"
        ),
    )
    parser.add_argument(
        "--num_vec_per_token",
        type=int,
        default=1,
        help=(
            "The number of vectors used to represent the placeholder token. The higher the number, the better the"
            " result at the cost of editability. This can be fixed by prompt editing."
        ),
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
        default=2,
        help="The slice amount for less memory usage. Higher slice means less memory but longer computation.",
    )
    parser.add_argument(
        "--project_name",
        type=str,
        default="huggingface_textual_inv",
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
        "--data_aug", action="store_true", help="Whether to do data augmentation before resizing to resolution"
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


# imagenet_templates_small = [
#     "a photo of a {}",
#     "a rendering of a {}",
#     "a cropped photo of the {}",
#     "the photo of a {}",
#     "a photo of a clean {}",
#     "a photo of a dirty {}",
#     "a dark photo of the {}",
#     "a photo of my {}",
#     "a photo of the cool {}",
#     "a close-up photo of a {}",
#     "a bright photo of the {}",
#     "a cropped photo of a {}",
#     "a photo of the {}",
#     "a good photo of the {}",
#     "a photo of one {}",
#     "a close-up photo of the {}",
#     "a rendition of the {}",
#     "a photo of the clean {}",
#     "a rendition of a {}",
#     "a photo of a nice {}",
#     "a good photo of a {}",
#     "a photo of the nice {}",
#     "a photo of the small {}",
#     "a photo of the weird {}",
#     "a photo of the large {}",
#     "a photo of a cool {}",
#     "a photo of a small {}",
# ]
imagenet_templates_small = [
    "a photo of a {}",
    "a picture of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
    "a photo of my {}",
    "a close-up photo of a {}",
    "a bright photo of the {}",
    "a cropped photo of a {}",
    "a photo of the {}",
    "a good photo of the {}",
    "a photo of one {}",
    "a close-up photo of the {}",
    "a good photo of a {}",
    "a photo of the nice {}"
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
        neg_placeholder_token="*",
        center_crop=False,
        data_aug=False,
    ):
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.learnable_property = learnable_property
        self.size = size
        self.placeholder_token = placeholder_token
        self.neg_placeholder_token = neg_placeholder_token
        self.center_crop = center_crop
        self.data_aug = data_aug
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
        resize_ratio = 0.9
        self.base_transform = A.Compose(
            [
                A.Rotate(p=0.5, limit=10, crop_border=True),
                A.RandomResizedCrop(self.size, self.size, scale=(resize_ratio, 1), ratio=(1, 1), p=0.5)
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
        text = random.choice(self.templates)

        example["input_ids"] = self.tokenizer(
            text.format(placeholder_string),
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]
        example["neg_input_ids"] = self.tokenizer(
            text.format(self.neg_placeholder_token),
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)
        if self.data_aug:
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
    print(f"accelerator device is {accelerator.device}")

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
            args.pretrained_model_name_or_path, subfolder="tokenizer"
        )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder"
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae"
    )
    # I found that the embeddings are made in an interesting way in dream artist by having the distributions be
    # visibly different from the default ones. So for negative, they do torch.randn *1e-3. There might be a demand for
    # having the distribution of the new embedding be different
    if args.guess_initializer_token:
        #
        guessed_concept = predict_replacement_words(
            [os.path.join(args.train_data_dir, file_path) for file_path in os.listdir(args.train_data_dir)]
        )
        token_ids = tokenizer.encode(guessed_concept, add_special_tokens=False)
        print(f"Guessed concept is {guessed_concept} token ids are {token_ids}")
        
    else:
        token_ids = tokenizer.encode(args.initializer_token, add_special_tokens=False)
    placeholder_token, placeholder_token_ids = add_tokens_and_get_placeholder_token(
        args, token_ids, tokenizer, text_encoder, args.num_vec_per_token, args.placeholder_token
    )

    neg_placeholder_token, neg_placeholder_token_ids = add_tokens_and_get_placeholder_token(
        args, token_ids, tokenizer, text_encoder, args.num_vec_per_token, args.placeholder_token+'_neg', is_random=True
    )

    if args.subject_noun:
        placeholder_token = f"{placeholder_token} {args.subject_noun}"
        print(f"Final placeholder token is {placeholder_token}")
    # Load models and create wrapper for stable diffusion

    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet"
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
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
    )

    train_dataset = TextualInversionDataset(
        data_root=args.train_data_dir,
        tokenizer=tokenizer,
        size=args.resolution,
        placeholder_token=placeholder_token,
        neg_placeholder_token=neg_placeholder_token,
        repeats=args.repeats,
        learnable_property=args.learnable_property,
        center_crop=args.center_crop,
        data_aug=args.data_aug,
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
    text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        text_encoder, optimizer, train_dataloader, lr_scheduler
    )
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    torch.cuda.empty_cache()
    # Move vae and unet to device
    # vae.encoder.to(device=accelerator.device, dtype=weight_dtype)
    # vae.quant_conv.to(accelerator.device, dtype=weight_dtype)
    # vae.post_quant_conv.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
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
    original_token_embeds = text_encoder.get_input_embeddings().weight.data.detach().clone().to(accelerator.device)

    for epoch in range(args.num_train_epochs):
        text_encoder.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.autocast(), accelerator.accumulate(text_encoder):
                # Convert images to latent space
                with torch.no_grad():
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
                noisy_latents = torch.cat([noisy_latents] * 2)

                # Get the text embedding for conditioning
                cond_embedding = text_encoder(batch["input_ids"])[0]
                uncond_embedding = text_encoder(batch["neg_input_ids"])[0]

                text_embeddings = torch.cat([uncond_embedding, cond_embedding])
                # Predict the noise residual
                # print(noisy_latents.shape, text_embeddings.shape)
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states=text_embeddings).sample
                # print(noise_pred.shape)
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_text - noise_pred_uncond)
                loss = F.mse_loss(noise_pred, noise, reduction="none").mean([1, 2, 3]).mean()
                accelerator.backward(loss)

                # Zero out the gradients for all token embeddings except the newly added
                # embeddings for the concept, as we only want to optimize the concept embeddings
                if accelerator.num_processes > 1:
                    token_embeds = text_encoder.module.get_input_embeddings().weight
                else:
                    token_embeds = text_encoder.get_input_embeddings().weight
                # Get the index for tokens that we want to zero the grads for
                grad_mask = torch.arange(len(tokenizer)) != placeholder_token_ids[0]
                for i in range(1, len(placeholder_token_ids)):
                    grad_mask = grad_mask & (torch.arange(len(tokenizer)) != placeholder_token_ids[i])
                for i in range(1, len(neg_placeholder_token_ids)):
                    grad_mask = grad_mask & (torch.arange(len(tokenizer)) != neg_placeholder_token_ids[i])
                token_embeds.data[grad_mask, :] = original_token_embeds[grad_mask, :]

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                del noise
                del noisy_latents
                del timesteps
                del cond_embedding
                del uncond_embedding
                del text_embeddings
                del noise_pred

                gc.collect()
                torch.cuda.empty_cache()
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                # Adding back weight decay
                progress_bar.update(1)
                global_step += 1
                if global_step % args.log_frequency == 0:
                    pipeline = get_pipeline(text_encoder, vae, unet, tokenizer, accelerator, weight_dtype)
                    log_progress(pipeline, args, global_step, wandb_run, placeholder_token, neg_placeholder_token)
                    if global_step % args.save_frequency == 0:
                        save_progress(
                            text_encoder, placeholder_token_ids, accelerator, args, placeholder_token, global_step
                        )

                    del pipeline
                    del grad_mask


            if global_step >= args.max_train_steps:
                break
            logs = {
                "loss": loss.detach().cpu().item() * args.gradient_accumulation_steps,
                "lr": lr_scheduler.get_last_lr()[0],
            }
            wandb_run.log(logs)
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            del loss
            if global_step >= args.max_train_steps:
                break

        accelerator.wait_for_everyone()

    # Create the pipeline using using the trained modules and save it.
    if accelerator.is_main_process:
        pipeline = get_pipeline(text_encoder, vae, unet, tokenizer, accelerator, weight_dtype)
        log_progress(pipeline, args, global_step, wandb_run, placeholder_token, neg_placeholder_token)
        pipeline.save_pretrained(args.output_dir)
        save_progress(text_encoder, placeholder_token_ids, accelerator, args, placeholder_token, global_step)
        # Also save the newly trained embeddings

        if args.push_to_hub:
            repo.push_to_hub(commit_message="End of training", blocking=False, auto_lfs_prune=True)

    accelerator.end_training()


if __name__ == "__main__":
    main()
