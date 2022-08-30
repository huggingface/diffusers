import argparse
import math
import os
import random
from typing import Any, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.utils.data import Dataset

import PIL
from accelerate import Accelerator
from accelerate.logging import get_logger
from datasets import load_dataset
from diffusers import AutoencoderKL, DDPMScheduler, PNDMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.hub_utils import init_git_repo, push_to_hub
from diffusers.optimization import get_scheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPFeatureExtractor, CLIPPreTrainedModel, CLIPTextConfig, CLIPTextModel, CLIPTokenizer
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.clip.modeling_clip import CLIPEncoder, CLIPTextEmbeddings, _expand_mask


class CLIPTextTransformer(nn.Module):
    def __init__(self, config: CLIPTextConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        self.embeddings = CLIPTextEmbeddings(config)
        self.encoder = CLIPEncoder(config)
        self.final_layer_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        inputs_embeds: Optional[torch.FloatTensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        input_shape = inputs_embeds.size()[:-1]
        hidden_states = self.embeddings(inputs_embeds=inputs_embeds, position_ids=position_ids)

        bsz, seq_len = input_shape
        # CLIP's text model uses causal mask, prepare it here.
        # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
        causal_attention_mask = self._build_causal_attention_mask(bsz, seq_len, hidden_states.dtype).to(
            hidden_states.device
        )
        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, hidden_states.dtype)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.final_layer_norm(last_hidden_state)

        # can't compute pooled_output without input_ids and it's need needed here.
        pooled_output = None

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def _build_causal_attention_mask(self, bsz, seq_len, dtype):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(bsz, seq_len, seq_len, dtype=dtype)
        mask.fill_(torch.tensor(torch.finfo(dtype).min))
        mask.triu_(1)  # zero out the lower diagonal
        mask = mask.unsqueeze(1)  # expand mask
        return mask


class CLIPTextualInversionWrapper(CLIPPreTrainedModel):
    config_class = CLIPTextConfig

    def __init__(self, config: CLIPTextConfig, placeholder_token_id: int, initializer_token_id: int):
        super().__init__(config)
        self.text_model = CLIPTextTransformer(config)

        # TODO (patil-suraj): generalize this to n tokens
        self.concept_embeddings = nn.Parameter(torch.zeros(config.hidden_size))

        self.placeholder_token_id = placeholder_token_id
        self.initializer_token_id = initializer_token_id

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.text_model.embeddings.token_embedding

    def set_input_embeddings(self, value):
        self.text_model.embeddings.token_embedding = value

    def freeze_text_model(self):
        for param in self.text_model.parameters():
            param.requires_grad = False

    def init_concept_embeddings(self):
        """Initialize concept embeddings with embeddings of initializer_token_id"""
        self.concept_embeddings.data.copy_(self.get_input_embeddings().weight[self.initializer_token_id])

    def merge_concept_embeddings_in_token_embeddings(self):
        """Update the token embeddings of the placeholder_token to the trained embeddings."""
        self.get_input_embeddings().weight.data[self.placeholder_token_id] = self.concept_embeddings.data

    def get_concept_embeddings(self):
        return self.concept_embeddings

    def get_initializer_embeddings(self):
        return self.get_input_embeddings().weight[self.initializer_token_id]

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:

        inputs_embeds = self.get_input_embeddings()(input_ids)
        # update the inputs_embeds for concept token embedding
        placeholder_idx = torch.where(input_ids == self.placeholder_token_id)
        inputs_embeds[placeholder_idx] = self.concept_embeddings

        return self.text_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


#############################################################


# Simple wrapper module for Stabel Diffusion
class StableDiffusionWrapper(nn.Module):
    def __init__(self, text_encoder: CLIPTextualInversionWrapper, vae: AutoencoderKL, unet: UNet2DConditionModel):
        super().__init__()
        self.text_encoder = text_encoder
        self.vae = vae
        self.unet = unet

    def freeze_text_encoder(self):
        self.text_encoder.freeze_text_model()

    def freeze_vae(self):
        for param in self.vae.parameters():
            param.requires_grad = False

    def freeze_unet(self):
        for param in self.unet.parameters():
            param.requires_grad = False


#############################################################
imagenet_templates_smallest = [
    "a photo of a {}",
]

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

imagenet_dual_templates_small = [
    "a photo of a {} with {}",
    "a rendering of a {} with {}",
    "a cropped photo of the {} with {}",
    "the photo of a {} with {}",
    "a photo of a clean {} with {}",
    "a photo of a dirty {} with {}",
    "a dark photo of the {} with {}",
    "a photo of my {} with {}",
    "a photo of the cool {} with {}",
    "a close-up photo of a {} with {}",
    "a bright photo of the {} with {}",
    "a cropped photo of a {} with {}",
    "a photo of the {} with {}",
    "a good photo of the {} with {}",
    "a photo of one {} with {}",
    "a close-up photo of the {} with {}",
    "a rendition of the {} with {}",
    "a photo of the clean {} with {}",
    "a rendition of a {} with {}",
    "a photo of a nice {} with {}",
    "a good photo of a {} with {}",
    "a photo of the nice {} with {}",
    "a photo of the small {} with {}",
    "a photo of the weird {} with {}",
    "a photo of the large {} with {}",
    "a photo of a cool {} with {}",
    "a photo of a small {} with {}",
]

per_img_token_list = [
    "א",
    "ב",
    "ג",
    "ד",
    "ה",
    "ו",
    "ז",
    "ח",
    "ט",
    "י",
    "כ",
    "ל",
    "מ",
    "נ",
    "ס",
    "ע",
    "פ",
    "צ",
    "ק",
    "ר",
    "ש",
    "ת",
]


class TextualInversionDataset(Dataset):
    def __init__(
        self,
        data_root,
        tokenizer,
        style="style",  # [concept, style] TODO: better names ?
        size=None,
        repeats=100,
        interpolation="bicubic",
        flip_p=0.5,
        set="train",
        placeholder_token="*",
        per_image_tokens=False,
        center_crop=False,
        mixing_prob=0.25,
        coarse_class_text=None,
    ):

        self.data_root = data_root
        self.tokenizer = tokenizer
        self.style = style

        self.image_paths = [os.path.join(self.data_root, file_path) for file_path in os.listdir(self.data_root)]

        # self._length = len(self.image_paths)
        self.num_images = len(self.image_paths)
        self._length = self.num_images

        self.placeholder_token = placeholder_token

        self.per_image_tokens = per_image_tokens
        self.center_crop = center_crop
        self.mixing_prob = mixing_prob

        self.coarse_class_text = coarse_class_text

        if per_image_tokens:
            assert self.num_images < len(per_img_token_list), (
                "Can't use per-image tokens when the training set contains more than"
                f" {len(per_img_token_list)} tokens. To enable larger sets, add more tokens to 'per_img_token_list'."
            )

        if set == "train":
            self._length = self.num_images * repeats

        self.size = size
        self.interpolation = {
            "linear": PIL.Image.LINEAR,
            "bilinear": PIL.Image.BILINEAR,
            "bicubic": PIL.Image.BICUBIC,
            "lanczos": PIL.Image.LANCZOS,
        }[interpolation]
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)

        self.templates = imagenet_style_templates_small if style == "style" else imagenet_templates_small

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        image = Image.open(self.image_paths[i % self.num_images])

        if not image.mode == "RGB":
            image = image.convert("RGB")

        placeholder_string = self.placeholder_token
        if self.coarse_class_text:
            placeholder_string = f"{self.coarse_class_text} {placeholder_string}"

        if self.per_image_tokens and np.random.uniform() < self.mixing_prob:
            text = random.choice(imagenet_dual_templates_small).format(
                placeholder_string, per_img_token_list[i % self.num_images]
            )
        else:
            text = random.choice(self.templates).format(placeholder_string)

        example["input_ids"] = self.tokenizer(
            text, padding="max_length", truncation=True, max_length=77, return_tensors="pt"
        ).input_ids

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)

        if self.center_crop:
            crop = min(img.shape[0], img.shape[1])
            h, w, = (
                img.shape[0],
                img.shape[1],
            )
            img = img[(h - crop) // 2 : (h + crop) // 2, (w - crop) // 2 : (w + crop) // 2]

        image = Image.fromarray(img)
        if self.size is not None:
            image = image.resize((self.size, self.size), resample=self.interpolation)

        image = self.flip(image)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)

        example["pixel_values"] = torch.from_numpy(image).permute(2, 0, 1)
        return example


def main(args):
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        logging_dir=logging_dir,
    )

    # Load the tokenizer and add the placeholder token as a additional special token
    if args.tokenizer_name:
        tokenizer = CLIPTokenizer.from_pretrained(
            args.tokenizer_name, additional_special_tokens=[args.placeholder_token]
        )
    elif args.pretrained_model_name_or_path:
        tokenizer = CLIPTokenizer.from_pretrained(
            args.model_name_or_path, additional_special_tokens=[args.placeholder_token], subfolder="tokenizer"
        )

    # Convert the initializer_token, placeholder_token to ids
    token_ids = tokenizer.encode(args.initializer_token, add_special_tokens=False)
    # Check if initializer_token is a single token or a sequence of tokens
    if len(token_ids) > 1:
        raise ValueError("The initializer token must be a single token.")

    initializer_token_id = token_ids[0]
    placeholder_token_id = tokenizer.convert_tokens_to_ids(args.placeholder_token)

    # Load models and create wrapper for stable diffusion
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    text_encoder = CLIPTextualInversionWrapper.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        placeholder_token_id=placeholder_token_id,
        initializer_token_id=initializer_token_id,
    )

    # Resize the token embeddings as we are adding new special tokens to the tokenizer
    text_encoder.resize_token_embeddings(len(tokenizer))

    # Initialize the new concept embeddings with the embeddings of the initializer token
    text_encoder.init_concept_embeddings()

    # Create a wrapper module for Stable Diffusion
    model = StableDiffusionWrapper(text_encoder, vae, unet)

    # Freeze everything except the concept embedding
    model.freeze_text_encoder()
    model.freeze_vae()
    model.freeze_unet()

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(
        [next(model.parameters())],  # only optimize the concept embeddings
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # TODO (patil-suraj): laod scheduler using args
    noise_scheduler = DDPMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000, tensor_format="pt"
    )

    dataset = TextualInversionDataset(
        data_root=args.train_data_dir,
        tokenizer=tokenizer,
        size=args.resolution,
        placeholder_token=args.placeholder_token,
        repeats=args.repeats,
        style=args.style,
        center_crop=args.center_crop,
        set="train",
    )
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if args.push_to_hub:
        repo = init_git_repo(args, at_init=True)

    if accelerator.is_main_process:
        run = os.path.split(__file__)[-1].split(".")[0]
        accelerator.init_trackers(run)

    global_step = 0
    for epoch in range(args.num_epochs):
        model.train()
        progress_bar = tqdm(total=num_update_steps_per_epoch, disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                # Convert images to latent space
                latents = model.vae.encode(batch["pixel_values"]).sample().detach()
                latents = latents * 0.18215

                # Sample noise that we'll add to the latents
                noise = torch.randn(latents.shape).to(latents.device)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device).long()

                # Add noise to t1`he latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # get the text embedding for conditioning
                input_ids = batch["input_ids"].reshape(bsz, -1)
                encoder_hidden_states = model.text_encoder(input_ids)[0]

                # Predict the noise residual
                noise_pred = model.unet(noisy_latents, timesteps, encoder_hidden_states)["sample"]
                loss = F.mse_loss(noise_pred, noise, reduction="none").mean([1, 2, 3]).mean()

                # Compute coarse loss
                # TODO: Choose better name for embedding_reg_weight
                if args.embedding_reg_weight > 0:
                    num_embeddings = 1  # TODO: generalize to multiple embeddings
                    optimized = model.text_encoder.get_concept_embeddings()
                    coarse = model.text_encoder.get_initializer_embeddings().clone().to(optimized.device)
                    coarse_loss = (optimized - coarse) @ (optimized - coarse).T / num_embeddings

                    loss += args.embedding_reg_weight * coarse_loss

                accelerator.backward(loss)
                # accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break
        progress_bar.close()

        accelerator.wait_for_everyone()

        # Generate sample images for visual inspection
        # if accelerator.is_main_process:
        #     if epoch % args.save_images_epochs == 0 or epoch == args.num_epochs - 1:
        #         pipeline = StableDiffusionPipeline(
        #             unet=accelerator.unwrap_model(stable_diffusion.unet),
        #             vae=accelerator.unwrap_model(stable_diffusion.vae),
        #             text_encoder=accelerator.unwrap_model(stable_diffusion.text_encoder),
        #             tokenizer=tokenizer,
        #             scheduler=noise_scheduler,
        #             safety_checker=StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker"),
        #             feature_extractor=CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32")
        #         )

        #         generator = torch.manual_seed(0)
        #         # run pipeline in inference (sample random noise and denoise)
        #         images = pipeline(generator=generator, batch_size=args.eval_batch_size, output_type="numpy")["sample"]

        #         # denormalize the images and save to tensorboard
        #         images_processed = (images * 255).round().astype("uint8")
        #         accelerator.trackers[0].writer.add_images(
        #             "test_samples", images_processed.transpose(0, 3, 1, 2), epoch
        #         )

        #     if epoch % args.save_model_epochs == 0 or epoch == args.num_epochs - 1:
        #         # save the model
        #         if args.push_to_hub:
        #             push_to_hub(args, pipeline, repo, commit_message=f"Epoch {epoch}", blocking=False)
        #         else:
        #             pipeline.save_pretrained(args.output_dir)
        accelerator.wait_for_everyone()

    # Create the pipeline using using the trained modules and save it.
    if accelerator.is_main_process:
        # Load the CLIPTextualInversionWrapper into CLIPTextModel.
        text_model = accelerator.unwrap_model(model.text_encoder)
        # Update the token embeddings of the placeholder_token to the trained embeddings.
        text_model.merge_concept_embeddings_in_token_embeddings()
        clip = CLIPTextModel(text_model.config)
        clip.load_state_dict(text_model.state_dict(), strict=False)

        pipeline = StableDiffusionPipeline(
            unet=accelerator.unwrap_model(model.unet),
            vae=accelerator.unwrap_model(model.vae),
            text_encoder=clip,
            tokenizer=tokenizer,
            scheduler=PNDMScheduler.from_config(
                "CompVis/stable-diffusion-v1-4", subfolder="scheduler", use_auth_token=True
            ),
            safety_checker=StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker"),
            feature_extractor=CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32"),
        )
        pipeline.save_pretrained(args.output_dir)

    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
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
    parser.add_argument("--style", type=str, default="concept", help="Choose between 'concept' and 'style'")
    parser.add_argument("--repeats", type=int, default=100, help="How many times to repeat the training data.")
    parser.add_argument("--embedding_reg_weight", type=float, default=0.0)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="text-inversion-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
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
    parser.add_argument(
        "--eval_batch_size", type=int, default=16, help="Batch size (per device) for the evaluation dataloader."
    )
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=5000,
        help="Total number of training steps to perform.",
    )
    parser.add_argument("--save_images_epochs", type=int, default=10)
    parser.add_argument("--save_model_epochs", type=int, default=10)
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
        "--hub_private_repo", action="store_true", help="Whether the model repository is private or not."
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

    main(args)
