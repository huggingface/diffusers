"""
Copyright (c) Alibaba, Inc. and its affiliates.
"""
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


# Copied from diffusers.models.controlnet.zero_module
def zero_module(module: nn.Module) -> nn.Module:
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


def get_clip_token_for_string(tokenizer, string):
    batch_encoding = tokenizer(
        string,
        truncation=True,
        max_length=77,
        return_length=True,
        return_overflowing_tokens=False,
        padding="max_length",
        return_tensors="pt",
    )
    tokens = batch_encoding["input_ids"]
    assert (
        torch.count_nonzero(tokens - 49407) == 2
    ), f"String '{string}' maps to more than a single token. Please use another string"
    return tokens[0, 1]


def get_bert_token_for_string(tokenizer, string):
    token = tokenizer(string)
    assert (
        torch.count_nonzero(token) == 3
    ), f"String '{string}' maps to more than a single token. Please use another string"
    token = token[0, 1]
    return token


def get_clip_vision_emb(encoder, processor, img):
    _img = img.repeat(1, 3, 1, 1) * 255
    inputs = processor(images=_img, return_tensors="pt")
    inputs["pixel_values"] = inputs["pixel_values"].to(img.device)
    outputs = encoder(**inputs)
    emb = outputs.image_embeds
    return emb


def get_recog_emb(encoder, img_list):
    _img_list = [(img.repeat(1, 3, 1, 1) * 255)[0] for img in img_list]
    encoder.predictor.eval()
    _, preds_neck = encoder.pred_imglist(_img_list, show_debug=False)
    return preds_neck


def pad_H(x):
    _, _, H, W = x.shape
    p_top = (W - H) // 2
    p_bot = W - H - p_top
    return F.pad(x, (0, 0, p_top, p_bot))


class EncodeNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncodeNet, self).__init__()
        chan = 16
        n_layer = 4  # downsample

        self.conv1 = conv_nd(2, in_channels, chan, 3, padding=1)
        self.conv_list = nn.ModuleList([])
        _c = chan
        for i in range(n_layer):
            self.conv_list.append(conv_nd(2, _c, _c * 2, 3, padding=1, stride=2))
            _c *= 2
        self.conv2 = conv_nd(2, _c, out_channels, 3, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.act(self.conv1(x))
        for layer in self.conv_list:
            x = self.act(layer(x))
        x = self.act(self.conv2(x))
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


class EmbeddingManager(nn.Module):
    def __init__(
        self,
        embedder,
        valid=True,
        glyph_channels=20,
        position_channels=1,
        placeholder_string="*",
        add_pos=False,
        emb_type="ocr",
        **kwargs,
    ):
        super().__init__()
        if hasattr(embedder, "tokenizer"):  # using Stable Diffusion's CLIP encoder
            get_token_for_string = partial(get_clip_token_for_string, embedder.tokenizer)
            token_dim = 768
            if hasattr(embedder, "vit"):
                assert emb_type == "vit"
                self.get_vision_emb = partial(get_clip_vision_emb, embedder.vit, embedder.processor)
            self.get_recog_emb = None
        else:  # using LDM's BERT encoder
            get_token_for_string = partial(get_bert_token_for_string, embedder.tknz_fn)
            token_dim = 1280
        self.token_dim = token_dim
        self.emb_type = emb_type

        self.add_pos = add_pos
        if add_pos:
            self.position_encoder = EncodeNet(position_channels, token_dim)
        if emb_type == "ocr":
            self.proj = nn.Sequential(zero_module(nn.Linear(40 * 64, token_dim)), nn.LayerNorm(token_dim))
        if emb_type == "conv":
            self.glyph_encoder = EncodeNet(glyph_channels, token_dim)

        self.placeholder_token = get_token_for_string(placeholder_string)

    def encode_text(self, text_info):
        if self.get_recog_emb is None and self.emb_type == "ocr":
            self.get_recog_emb = partial(get_recog_emb, self.recog)

        gline_list = []
        pos_list = []
        for i in range(len(text_info["n_lines"])):  # sample index in a batch
            n_lines = text_info["n_lines"][i]
            for j in range(n_lines):  # line
                gline_list += [text_info["gly_line"][j][i : i + 1]]
                if self.add_pos:
                    pos_list += [text_info["positions"][j][i : i + 1]]

        if len(gline_list) > 0:
            if self.emb_type == "ocr":
                recog_emb = self.get_recog_emb(gline_list)
                enc_glyph = self.proj(recog_emb.reshape(recog_emb.shape[0], -1))
            elif self.emb_type == "vit":
                enc_glyph = self.get_vision_emb(pad_H(torch.cat(gline_list, dim=0)))
            elif self.emb_type == "conv":
                enc_glyph = self.glyph_encoder(pad_H(torch.cat(gline_list, dim=0)))
            if self.add_pos:
                enc_pos = self.position_encoder(torch.cat(gline_list, dim=0))
                enc_glyph = enc_glyph + enc_pos

        self.text_embs_all = []
        n_idx = 0
        for i in range(len(text_info["n_lines"])):  # sample index in a batch
            n_lines = text_info["n_lines"][i]
            text_embs = []
            for j in range(n_lines):  # line
                text_embs += [enc_glyph[n_idx : n_idx + 1]]
                n_idx += 1
            self.text_embs_all += [text_embs]

    def forward(
        self,
        tokenized_text,
        embedded_text,
    ):
        b, device = tokenized_text.shape[0], tokenized_text.device
        for i in range(b):
            idx = tokenized_text[i] == self.placeholder_token.to(device)
            if sum(idx) > 0:
                if i >= len(self.text_embs_all):
                    print("truncation for log images...")
                    break
                text_emb = torch.cat(self.text_embs_all[i], dim=0)
                if sum(idx) != len(text_emb):
                    print("truncation for long caption...")
                embedded_text[i][idx] = text_emb[: sum(idx)]
        return embedded_text

    def embedding_parameters(self):
        return self.parameters()
