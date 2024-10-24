"""
Copyright (c) Alibaba, Inc. and its affiliates.
"""
from functools import partial

import torch
import torch.nn as nn
from safetensors.torch import load_file


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


def get_recog_emb(encoder, img_list):
    _img_list = [(img.repeat(1, 3, 1, 1) * 255)[0] for img in img_list]
    encoder.predictor.eval()
    _, preds_neck = encoder.pred_imglist(_img_list, show_debug=False)
    return preds_neck


class EmbeddingManager(nn.Module):
    def __init__(
        self,
        embedder,
        placeholder_string="*",
        use_fp16=False,
    ):
        super().__init__()
        get_token_for_string = partial(get_clip_token_for_string, embedder.tokenizer)
        token_dim = 768
        self.get_recog_emb = None
        self.token_dim = token_dim

        self.proj = nn.Linear(40 * 64, token_dim)
        self.proj.load_state_dict(load_file("proj.safetensors", device=str(embedder.device)))
        if use_fp16:
            self.proj = self.proj.to(dtype=torch.float16)

        self.placeholder_token = get_token_for_string(placeholder_string)

    @torch.no_grad()
    def encode_text(self, text_info):
        if self.get_recog_emb is None:
            self.get_recog_emb = partial(get_recog_emb, self.recog)

        gline_list = []
        for i in range(len(text_info["n_lines"])):  # sample index in a batch
            n_lines = text_info["n_lines"][i]
            for j in range(n_lines):  # line
                gline_list += [text_info["gly_line"][j][i : i + 1]]

        if len(gline_list) > 0:
            recog_emb = self.get_recog_emb(gline_list)
            enc_glyph = self.proj(recog_emb.reshape(recog_emb.shape[0], -1).to(self.proj.weight.dtype))

        self.text_embs_all = []
        n_idx = 0
        for i in range(len(text_info["n_lines"])):  # sample index in a batch
            n_lines = text_info["n_lines"][i]
            text_embs = []
            for j in range(n_lines):  # line
                text_embs += [enc_glyph[n_idx : n_idx + 1]]
                n_idx += 1
            self.text_embs_all += [text_embs]

    @torch.no_grad()
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
                text_emb = text_emb.to(embedded_text.device)
                embedded_text[i][idx] = text_emb[: sum(idx)]
        return embedded_text

    def embedding_parameters(self):
        return self.parameters()
