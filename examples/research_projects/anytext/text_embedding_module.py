# text -> glyph render -> glyph lines -> OCR -> linear ->
# +> Token Replacement -> FrozenCLIPEmbedderT3
# text -> tokenizer ->


import torch
from PIL import ImageFont
from torch import nn

from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.utils import logging

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
    def forward(self, prompt, device, num_images_per_prompt, do_classifier_free_guidance, hint, n_prompt, text_info):
        prompt_embeds = self.get_learned_conditioning(
            {"c_concat": [hint], "c_crossattn": [[prompt] * len(prompt)], "text_info": text_info}
        )
        negative_prompt_embeds = self.get_learned_conditioning(
            {"c_concat": [hint], "c_crossattn": [[n_prompt] * len(prompt)], "text_info": text_info}
        )

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

    def get_unconditional_conditioning(self, N):
        return self.get_learned_conditioning({"c_crossattn": [[""] * N], "text_info": None})
