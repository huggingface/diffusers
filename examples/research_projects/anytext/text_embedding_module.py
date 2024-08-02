# text -> glyph render -> glyph lines -> OCR -> linear ->
# +> Token Replacement -> FrozenCLIPEmbedderT3
# text -> tokenizer ->

from typing import List, Optional

import cv2
import numpy as np
import torch
from easydict import EasyDict as edict
from PIL import Image, ImageDraw, ImageFont
from torch import nn

from diffusers.loaders import StableDiffusionLoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers

from .embedding_manager import EmbeddingManager
from .frozen_clip_embedder_t3 import FrozenCLIPEmbedderT3
from .recognizer import TextRecognizer, create_predictor


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class TextEmbeddingModule(nn.Module):
    def __init__(self, font_path, device):
        super().__init__()
        self.device = device
        self.font = ImageFont.truetype(font_path, 60)
        self.ocr_model = ...
        self.linear = nn.Linear()
        self.frozen_CLIP_embedder_t3 = FrozenCLIPEmbedderT3(device=self.device)
        self.embedding_manager_config = {
            "valid": True,
            "emb_type": "ocr",
            "glyph_channels": 1,
            "position_channels": 1,
            "add_pos": False,
            "placeholder_string": "*",
        }
        self.embedding_manager = EmbeddingManager(self.frozen_CLIP_embedder_t3, **self.embedding_manager_config)
        # TODO: Understand the reason of param.requires_grad = True
        for param in self.embedding_manager.embedding_parameters():
            param.requires_grad = True
        rec_model_dir = "./ocr_weights/ppv3_rec.pth"
        self.text_predictor = create_predictor(rec_model_dir).eval()
        args = edict()
        args.rec_image_shape = "3, 48, 320"
        args.rec_batch_num = 6
        args.rec_char_dict_path = "./ocr_recog/ppocr_keys_v1.txt"
        args.use_fp16 = self.use_fp16
        self.cn_recognizer = TextRecognizer(args, self.text_predictor)
        for param in self.text_predictor.parameters():
            param.requires_grad = False
        self.embedding_manager.recog = self.cn_recognizer

    @torch.no_grad()
    def forward(self, texts, prompt, device, num_images_per_prompt, do_classifier_free_guidance):
        glyph_lines = self.create_glyph_lines(texts)
        ocr_output = self.ocr(glyph_lines)
        _ = self.linear(ocr_output)
        # Token Replacement

        # FrozenCLIPEmbedderT3
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt=None,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            lora_scale=None,
            clip_skip=None,
        )
        return prompt_embeds, negative_prompt_embeds

    def ocr(self, glyph_lines):
        pass

    def create_glyph_lines(
        self,
        texts,
        mode="text-generation",
        img_count=1,
        max_chars=77,
        draw_pos=None,
        ori_image=None,
        sort_priority=False,
        h=512,
        w=512,
    ):
        if mode in ["text-generation", "gen"]:
            edit_image = np.ones((h, w, 3)) * 127.5  # empty mask image
        elif mode in ["text-editing", "edit"]:
            if isinstance(ori_image, str):
                ori_image = cv2.imread(ori_image)[..., ::-1]
            elif isinstance(ori_image, torch.Tensor):
                ori_image = ori_image.cpu().numpy()
            edit_image = ori_image.clip(1, 255)  # for mask reason
            edit_image = self.check_channels(edit_image)
            edit_image = self.resize_image(
                edit_image, max_length=768
            )  # make w h multiple of 64, resize if w or h > max_length
            h, w = edit_image.shape[:2]  # change h, w by input ref_img
        # preprocess pos_imgs(if numpy, make sure it's white pos in black bg)
        if draw_pos is None:
            pos_imgs = np.zeros((w, h, 1))
        if isinstance(draw_pos, str):
            draw_pos = cv2.imread(draw_pos)[..., ::-1]
            pos_imgs = 255 - draw_pos
        elif isinstance(draw_pos, torch.Tensor):
            pos_imgs = draw_pos.cpu().numpy()
        if mode in ["text-editing", "edit"]:
            pos_imgs = cv2.resize(pos_imgs, (w, h))
        pos_imgs = pos_imgs[..., 0:1]
        pos_imgs = cv2.convertScaleAbs(pos_imgs)
        _, pos_imgs = cv2.threshold(pos_imgs, 254, 255, cv2.THRESH_BINARY)
        # separate pos_imgs
        pos_imgs = self.separate_pos_imgs(pos_imgs, sort_priority)
        if len(pos_imgs) == 0:
            pos_imgs = [np.zeros((h, w, 1))]
        # get pre_pos that needed for anytext
        pre_pos = []
        for input_pos in pos_imgs:
            if input_pos.mean() != 0:
                input_pos = input_pos[..., np.newaxis] if len(input_pos.shape) == 2 else input_pos
                poly, pos_img = self.find_polygon(input_pos)
                pre_pos += [pos_img / 255.0]
            else:
                pre_pos += [np.zeros((h, w, 1))]
        # prepare info dict
        gly_lines = []
        for i in range(len(texts)):
            text = texts[i]
            if len(text) > max_chars:
                text = text[:max_chars]
            if pre_pos[i].mean() != 0:
                gly_line = self.draw_glyph(self.font, text)
            else:
                gly_line = np.zeros((80, 512, 1))
            gly_lines += [self.arr2tensor(gly_line, img_count)]

        return gly_lines

    def check_channels(self, image):
        channels = image.shape[2] if len(image.shape) == 3 else 1
        if channels == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif channels > 3:
            image = image[:, :, :3]
        return image

    def resize_image(self, img, max_length=768):
        height, width = img.shape[:2]
        max_dimension = max(height, width)

        if max_dimension > max_length:
            scale_factor = max_length / max_dimension
            new_width = int(round(width * scale_factor))
            new_height = int(round(height * scale_factor))
            new_size = (new_width, new_height)
            img = cv2.resize(img, new_size)
        height, width = img.shape[:2]
        img = cv2.resize(img, (width - (width % 64), height - (height % 64)))
        return img

    def draw_glyph(self, font, text):
        g_size = 50
        W, H = (512, 80)
        new_font = font.font_variant(size=g_size)
        img = Image.new(mode="1", size=(W, H), color=0)
        draw = ImageDraw.Draw(img)
        left, top, right, bottom = new_font.getbbox(text)
        text_width = max(right - left, 5)
        text_height = max(bottom - top, 5)
        ratio = min(W * 0.9 / text_width, H * 0.9 / text_height)
        new_font = font.font_variant(size=int(g_size * ratio))

        text_width, text_height = new_font.getsize(text)
        offset_x, offset_y = new_font.getoffset(text)
        x = (img.width - text_width) // 2
        y = (img.height - text_height) // 2 - offset_y // 2
        draw.text((x, y), text, font=new_font, fill="white")
        img = np.expand_dims(np.array(img), axis=2).astype(np.float64)
        return img

    def encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        lora_scale: Optional[float] = None,
        clip_skip: Optional[int] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            lora_scale (`float`, *optional*):
                A LoRA scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
        """
        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, StableDiffusionLoraLoaderMixin):
            self._lora_scale = lora_scale

            # dynamically adjust the LoRA scale
            if not USE_PEFT_BACKEND:
                adjust_lora_scale_text_encoder(self.frozen_CLIP_embedder_t3.text_encoder, lora_scale)
            else:
                scale_lora_layers(self.frozen_CLIP_embedder_t3.text_encoder, lora_scale)

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            # textual inversion: process multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                prompt = self.maybe_convert_prompt(prompt, self.frozen_CLIP_embedder_t3.tokenizer)

            text_inputs = self.frozen_CLIP_embedder_t3.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.frozen_CLIP_embedder_t3.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.frozen_CLIP_embedder_t3.tokenizer(
                prompt, padding="longest", return_tensors="pt"
            ).input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = self.frozen_CLIP_embedder_t3.tokenizer.batch_decode(
                    untruncated_ids[:, self.frozen_CLIP_embedder_t3.tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.frozen_CLIP_embedder_t3.tokenizer.model_max_length} tokens: {removed_text}"
                )

            if (
                hasattr(self.frozen_CLIP_embedder_t3.text_encoder.config, "use_attention_mask")
                and self.frozen_CLIP_embedder_t3.text_encoder.config.use_attention_mask
            ):
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            if clip_skip is None:
                prompt_embeds = self.frozen_CLIP_embedder_t3.text_encoder(
                    text_input_ids.to(device), attention_mask=attention_mask
                )
                prompt_embeds = prompt_embeds[0]
            else:
                prompt_embeds = self.frozen_CLIP_embedder_t3.text_encoder(
                    text_input_ids.to(device), attention_mask=attention_mask, output_hidden_states=True
                )
                # Access the `hidden_states` first, that contains a tuple of
                # all the hidden states from the encoder layers. Then index into
                # the tuple to access the hidden states from the desired layer.
                prompt_embeds = prompt_embeds[-1][-(clip_skip + 1)]
                # We also need to apply the final LayerNorm here to not mess with the
                # representations. The `last_hidden_states` that we typically use for
                # obtaining the final prompt representations passes through the LayerNorm
                # layer.
                prompt_embeds = self.frozen_CLIP_embedder_t3.text_encoder.text_model.final_layer_norm(prompt_embeds)

        if self.text_encoder is not None:
            prompt_embeds_dtype = self.text_encoder.dtype
        elif self.unet is not None:
            prompt_embeds_dtype = self.unet.dtype
        else:
            prompt_embeds_dtype = prompt_embeds.dtype

        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            # textual inversion: process multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                uncond_tokens = self.maybe_convert_prompt(uncond_tokens, self.frozen_CLIP_embedder_t3.tokenizer)

            max_length = prompt_embeds.shape[1]
            uncond_input = self.frozen_CLIP_embedder_t3.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if (
                hasattr(self.frozen_CLIP_embedder_t3.text_encoder.config, "use_attention_mask")
                and self.frozen_CLIP_embedder_t3.text_encoder.config.use_attention_mask
            ):
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = self.frozen_CLIP_embedder_t3.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        if self.frozen_CLIP_embedder_t3.text_encoder is not None:
            if isinstance(self, StableDiffusionLoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(self.frozen_CLIP_embedder_t3.text_encoder, lora_scale)

        return prompt_embeds, negative_prompt_embeds
