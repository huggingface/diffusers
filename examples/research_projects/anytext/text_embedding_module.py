# text -> glyph render -> glyph lines -> OCR -> linear ->
# +> Token Replacement -> FrozenCLIPEmbedderT3
# text -> tokenizer ->


from typing import List, Optional

import torch
from PIL import ImageFont
from torch import nn

from diffusers.loaders import (
    StableDiffusionLoraLoaderMixin,
    TextualInversionLoaderMixin,
)
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.utils import (
    USE_PEFT_BACKEND,
    logging,
    scale_lora_layers,
    unscale_lora_layers,
)

from .embedding_manager import EmbeddingManager
from .frozen_clip_embedder_t3 import FrozenCLIPEmbedderT3
from .recognizer import TextRecognizer, create_predictor


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class TextEmbeddingModule(nn.Module):
    def __init__(self, font_path, device, use_fp16):
        super().__init__()
        self.device = device
        # TODO: Learn if the recommended font file is free to use
        self.font = ImageFont.truetype(font_path, 60)
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
        args = {}
        args["rec_image_shape"] = "3, 48, 320"
        args["rec_batch_num"] = 6
        args["rec_char_dict_path"] = "./ocr_recog/ppocr_keys_v1.txt"
        args["use_fp16"] = use_fp16
        self.cn_recognizer = TextRecognizer(args, self.text_predictor)
        for param in self.text_predictor.parameters():
            param.requires_grad = False
        self.embedding_manager.recog = self.cn_recognizer

    @torch.no_grad()
    def forward(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        hint,
        text_info,
        negative_prompt=None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        lora_scale: Optional[float] = None,
        clip_skip: Optional[int] = None,
    ):
        # TODO: Convert `get_learned_conditioning` functions to `diffusers`' format
        prompt_embeds = self.get_learned_conditioning(
            {"c_concat": [hint], "c_crossattn": [[prompt] * num_images_per_prompt], "text_info": text_info}
        )
        negative_prompt_embeds = self.get_learned_conditioning(
            {"c_concat": [hint], "c_crossattn": [[negative_prompt] * num_images_per_prompt], "text_info": text_info}
        )

        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, StableDiffusionLoraLoaderMixin):
            self._lora_scale = lora_scale

            # dynamically adjust the LoRA scale
            if not USE_PEFT_BACKEND:
                adjust_lora_scale_text_encoder(self.text_encoder, lora_scale)
            else:
                scale_lora_layers(self.text_encoder, lora_scale)

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            # textual inversion: process multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            if clip_skip is None:
                prompt_embeds = self.text_encoder(text_input_ids.to(device), attention_mask=attention_mask)
                prompt_embeds = prompt_embeds[0]
            else:
                prompt_embeds = self.text_encoder(
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
                prompt_embeds = self.text_encoder.text_model.final_layer_norm(prompt_embeds)

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
                uncond_tokens = self.maybe_convert_prompt(uncond_tokens, self.tokenizer)

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(
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

        if self.text_encoder is not None:
            if isinstance(self, StableDiffusionLoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(self.text_encoder, lora_scale)

        return prompt_embeds, negative_prompt_embeds

    def get_learned_conditioning(self, c):
        if hasattr(self.frozen_CLIP_embedder_t3, "encode") and callable(self.frozen_CLIP_embedder_t3.encode):
            if self.embedding_manager is not None and c["text_info"] is not None:
                self.embedding_manager.encode_text(c["text_info"])
            if isinstance(c, dict):
                cond_txt = c["c_crossattn"][0]
            else:
                cond_txt = c
            if self.embedding_manager is not None:
                cond_txt = self.frozen_CLIP_embedder_t3.encode(cond_txt, embedding_manager=self.embedding_manager)
            else:
                cond_txt = self.frozen_CLIP_embedder_t3.encode(cond_txt)
            if isinstance(c, dict):
                c["c_crossattn"][0] = cond_txt
            else:
                c = cond_txt
            if isinstance(c, DiagonalGaussianDistribution):
                c = c.mode()
        else:
            c = self.frozen_CLIP_embedder_t3(c)

        return c
