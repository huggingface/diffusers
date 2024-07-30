# text -> glyph render -> glyph lines -> OCR -> linear ->
# +> Token Replacement -> FrozenCLIPEmbedderT3
# text -> tokenizer ->

from typing import List, Optional

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn

from diffusers.loaders import StableDiffusionLoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers

from .frozen_clip_embedder_t3 import FrozenCLIPEmbedderT3


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def check_channels(image):
    channels = image.shape[2] if len(image.shape) == 3 else 1
    if channels == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif channels > 3:
        image = image[:, :, :3]
    return image


def resize_image(img, max_length=768):
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


def insert_spaces(string, nSpace):
    if nSpace == 0:
        return string
    new_string = ""
    for char in string:
        new_string += char + " " * nSpace
    return new_string[:-nSpace]


def draw_glyph(font, text):
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


def draw_glyph2(font, text, polygon, vertAng=10, scale=1, width=512, height=512, add_space=True):
    enlarge_polygon = polygon * scale
    rect = cv2.minAreaRect(enlarge_polygon)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    w, h = rect[1]
    angle = rect[2]
    if angle < -45:
        angle += 90
    angle = -angle
    if w < h:
        angle += 90

    vert = False
    if abs(angle) % 90 < vertAng or abs(90 - abs(angle) % 90) % 90 < vertAng:
        _w = max(box[:, 0]) - min(box[:, 0])
        _h = max(box[:, 1]) - min(box[:, 1])
        if _h >= _w:
            vert = True
            angle = 0

    img = np.zeros((height * scale, width * scale, 3), np.uint8)
    img = Image.fromarray(img)

    # infer font size
    image4ratio = Image.new("RGB", img.size, "white")
    draw = ImageDraw.Draw(image4ratio)
    _, _, _tw, _th = draw.textbbox(xy=(0, 0), text=text, font=font)
    text_w = min(w, h) * (_tw / _th)
    if text_w <= max(w, h):
        # add space
        if len(text) > 1 and not vert and add_space:
            for i in range(1, 100):
                text_space = insert_spaces(text, i)
                _, _, _tw2, _th2 = draw.textbbox(xy=(0, 0), text=text_space, font=font)
                if min(w, h) * (_tw2 / _th2) > max(w, h):
                    break
            text = insert_spaces(text, i - 1)
        font_size = min(w, h) * 0.80
    else:
        shrink = 0.75 if vert else 0.85
        font_size = min(w, h) / (text_w / max(w, h)) * shrink
    new_font = font.font_variant(size=int(font_size))

    left, top, right, bottom = new_font.getbbox(text)
    text_width = right - left
    text_height = bottom - top

    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    if not vert:
        draw.text(
            (rect[0][0] - text_width // 2, rect[0][1] - text_height // 2 - top),
            text,
            font=new_font,
            fill=(255, 255, 255, 255),
        )
    else:
        x_s = min(box[:, 0]) + _w // 2 - text_height // 2
        y_s = min(box[:, 1])
        for c in text:
            draw.text((x_s, y_s), c, font=new_font, fill=(255, 255, 255, 255))
            _, _t, _, _b = new_font.getbbox(c)
            y_s += _b

    rotated_layer = layer.rotate(angle, expand=1, center=(rect[0][0], rect[0][1]))

    x_offset = int((img.width - rotated_layer.width) / 2)
    y_offset = int((img.height - rotated_layer.height) / 2)
    img.paste(rotated_layer, (x_offset, y_offset), rotated_layer)
    img = np.expand_dims(np.array(img.convert("1")), axis=2).astype(np.float64)
    return img


class TextEmbeddingModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.frozen_CLIP_embedder_t3 = FrozenCLIPEmbedderT3()

    def forward(self, text):
        pass

    def create_glyph_lines(
        self,
        prompt,
        texts,
        mode="text-generation",
        img_count=1,
        max_chars=77,
        revise_pos=False,
        draw_pos=None,
        ori_image=None,
        sort_priority=False,
        h=512,
        w=512,
    ):
        if prompt is None and texts is None:
            raise ValueError("Prompt or texts should be provided!")
            # return None, -1, "You have input Chinese prompt but the translator is not loaded!", ""
        n_lines = len(texts)
        if mode in ["text-generation", "gen"]:
            edit_image = np.ones((h, w, 3)) * 127.5  # empty mask image
        elif mode in ["text-editing", "edit"]:
            if draw_pos is None or ori_image is None:
                raise ValueError("Reference image and position image are needed for text editing!")
                # return None, -1, "Reference image and position image are needed for text editing!", ""
            if isinstance(ori_image, str):
                ori_image = cv2.imread(ori_image)[..., ::-1]
                if ori_image is None:
                    raise ValueError(f"Can't read ori_image image from{ori_image}!")
                # assert ori_image is not None, f"Can't read ori_image image from{ori_image}!"
            elif isinstance(ori_image, torch.Tensor):
                ori_image = ori_image.cpu().numpy()
            else:
                if not isinstance(ori_image, np.ndarray):
                    raise ValueError(f"Unknown format of ori_image: {type(ori_image)}")
                # assert isinstance(ori_image, np.ndarray), f'Unknown format of ori_image: {type(ori_image)}'
            edit_image = ori_image.clip(1, 255)  # for mask reason
            edit_image = check_channels(edit_image)
            edit_image = resize_image(
                edit_image, max_length=768
            )  # make w h multiple of 64, resize if w or h > max_length
            h, w = edit_image.shape[:2]  # change h, w by input ref_img
        # preprocess pos_imgs(if numpy, make sure it's white pos in black bg)
        if draw_pos is None:
            pos_imgs = np.zeros((w, h, 1))
        if isinstance(draw_pos, str):
            draw_pos = cv2.imread(draw_pos)[..., ::-1]
            if draw_pos is None:
                raise ValueError(f"Can't read draw_pos image from{draw_pos}!")
            # assert draw_pos is not None, f"Can't read draw_pos image from{draw_pos}!"
            pos_imgs = 255 - draw_pos
        elif isinstance(draw_pos, torch.Tensor):
            pos_imgs = draw_pos.cpu().numpy()
        else:
            if not isinstance(draw_pos, np.ndarray):
                raise ValueError(f"Unknown format of draw_pos: {type(draw_pos)}")
            # assert isinstance(draw_pos, np.ndarray), f'Unknown format of draw_pos: {type(draw_pos)}'
        if mode in ["text-editing", "edit"]:
            pos_imgs = cv2.resize(pos_imgs, (w, h))
        pos_imgs = pos_imgs[..., 0:1]
        pos_imgs = cv2.convertScaleAbs(pos_imgs)
        _, pos_imgs = cv2.threshold(pos_imgs, 254, 255, cv2.THRESH_BINARY)
        # separate pos_imgs
        pos_imgs = self.separate_pos_imgs(pos_imgs, sort_priority)
        if len(pos_imgs) == 0:
            pos_imgs = [np.zeros((h, w, 1))]
        if len(pos_imgs) < n_lines:
            if n_lines == 1 and texts[0] == " ":
                pass  # text-to-image without text
            else:
                raise ValueError(
                    f"Found {len(pos_imgs)} positions that < needed {n_lines} from prompt, check and try again!"
                )
                # return None, -1, f'Found {len(pos_imgs)} positions that < needed {n_lines} from prompt, check and try again!', ''
        elif len(pos_imgs) > n_lines:
            str_warning = f"Warning: found {len(pos_imgs)} positions that > needed {n_lines} from prompt."
            logger.warning(str_warning)
        # get pre_pos, poly_list, hint that needed for anytext
        pre_pos = []
        poly_list = []
        for input_pos in pos_imgs:
            if input_pos.mean() != 0:
                input_pos = input_pos[..., np.newaxis] if len(input_pos.shape) == 2 else input_pos
                poly, pos_img = self.find_polygon(input_pos)
                pre_pos += [pos_img / 255.0]
                poly_list += [poly]
            else:
                pre_pos += [np.zeros((h, w, 1))]
                poly_list += [None]
        # prepare info dict
        info = {}
        info["gly_line"] = []
        gly_pos_imgs = []
        for i in range(len(texts)):
            text = texts[i]
            if len(text) > max_chars:
                str_warning = f'"{text}" length > max_chars: {max_chars}, will be cut off...'
                logger.warning(str_warning)
                text = text[:max_chars]
            gly_scale = 2
            if pre_pos[i].mean() != 0:
                gly_line = draw_glyph(self.font, text)
                glyphs = draw_glyph2(
                    self.font, text, poly_list[i], scale=gly_scale, width=w, height=h, add_space=False
                )
                gly_pos_img = cv2.drawContours(glyphs * 255, [poly_list[i] * gly_scale], 0, (255, 255, 255), 1)
                if revise_pos:
                    resize_gly = cv2.resize(glyphs, (pre_pos[i].shape[1], pre_pos[i].shape[0]))
                    new_pos = cv2.morphologyEx(
                        (resize_gly * 255).astype(np.uint8),
                        cv2.MORPH_CLOSE,
                        kernel=np.ones((resize_gly.shape[0] // 10, resize_gly.shape[1] // 10), dtype=np.uint8),
                        iterations=1,
                    )
                    new_pos = new_pos[..., np.newaxis] if len(new_pos.shape) == 2 else new_pos
                    contours, _ = cv2.findContours(new_pos, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    if len(contours) != 1:
                        str_warning = f"Fail to revise position {i} to bounding rect, remain position unchanged..."
                        logger.warning(str_warning)
                    else:
                        rect = cv2.minAreaRect(contours[0])
                        poly = np.int0(cv2.boxPoints(rect))
                        gly_pos_img = cv2.drawContours(glyphs * 255, [poly * gly_scale], 0, (255, 255, 255), 1)
                gly_pos_imgs += [gly_pos_img]  # for show
            else:
                gly_line = np.zeros((80, 512, 1))
                gly_pos_imgs += [np.zeros((h * gly_scale, w * gly_scale, 1))]  # for show
            info["gly_line"] += [self.arr2tensor(gly_line, img_count)]

        return info["gly_line"]

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
