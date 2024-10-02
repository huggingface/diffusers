import argparse
from typing import Any, Dict

import torch
from transformers import T5EncoderModel, T5Tokenizer

from diffusers import (
    CogView3PlusTransformer2DModel,
    CogView3PlusPipeline,
)

TRANSFORMER_KEYS_RENAME_DICT = {
    "transformer": "transformer_blocks",
    "attention": "attn1",
    "mlp": "ff.net",
    "dense_h_to_4h": "0.proj",
    "dense_4h_to_h": "2",
    ".layers": "",
    "dense": "to_out.0",
    "patch_embed": "norm1.norm",
    "post_attn1_layernorm": "norm2.norm",
    "mixins.patch_embed": "patch_embed",
    "mixins.final_layer.adaln": "norm_out",
    "mixins.final_layer.linear": "proj_out",
}