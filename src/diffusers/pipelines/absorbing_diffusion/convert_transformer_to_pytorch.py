# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Convert Absorbing Diffusion Transformer checkpoints from the original repository.

URL: https://github.com/samb-t/unleashing-transformers"""


import argparse
from pathlib import Path

import torch

from diffusers.models import Transformer
from huggingface_hub import HfApi


def remove_ignore_keys(state_dict):
    ignore_keys = [
        "Lt_history",
        "Lt_count",
        "loss_history",
    ]
    for k in ignore_keys:
        state_dict.pop(k, None)


def rename_key(name):
    if "_denoise_fn." in name:
        name = name.replace("_denoise_fn.", "")
    if "tok_emb" in name:
        name = name.replace("tok_emb", "token_embeddings")
    if "pos_emb" in name:
        name = name.replace("pos_emb", "position_embeddings")
    if "start_tok" in name:
        name = name.replace("start_tok", "start_token")
    if "attn" in name:
        name = name.replace("attn", "attention")
    if "ln1" in name:
        name = name.replace("ln1", "layernorm1")
    if "ln2" in name:
        name = name.replace("ln2", "layernorm2")
    if "proj" in name:
        name = name.replace("proj", "projection")
    if "ln_f" in name:
        name = name.replace("ln_f", "final_layernorm")

    return name


def convert_state_dict(orig_state_dict):
    for key in orig_state_dict.copy().keys():
        val = orig_state_dict.pop(key)

        if key.startswith("_denoise_fn"):
            orig_state_dict[rename_key(key)] = val
        else:
            pass

    return orig_state_dict


@torch.no_grad()
def convert_transformer_checkpoint(checkpoint_path, pytorch_dump_folder_path, push_to_hub=False):
    """
    Copy/paste/tweak model's weights to our Transformer structure.
    """
    # load original state dict, convert
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    remove_ignore_keys(state_dict)
    new_state_dict = convert_state_dict(state_dict)

    # load 🤗 model
    model = Transformer(vocab_size=1024)
    model.eval()

    # load weights
    model.load_state_dict(new_state_dict)

    # Verify outputs on dummy input
    dummy_input = torch.tensor([[1024, 1024, 1024]])
    logits = model(dummy_input).logits

    assert logits.shape == (1, 3, 1024)
    assert torch.allclose(
        logits[0, :3, :3],
        torch.tensor([[-0.1161, 0.4330, -13.3936], [-0.0973, 0.4164, -13.3616], [-0.1220, 0.4098, -13.3992]]),
        atol=1e-4,
    )

    print("Looks ok!")

    if pytorch_dump_folder_path is not None:
        Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
        print(f"Saving model to {pytorch_dump_folder_path}")
        model.save_pretrained(pytorch_dump_folder_path)

    if push_to_hub:
        # save the model
        pytorch_dump_folder_path = "/Users/nielsrogge/Documents/AbsorbingDiffusion/churches/test/transformer"
        model.save_pretrained(pytorch_dump_folder_path)

        # push to the hub
        api = HfApi()
        api.upload_folder(
            folder_path=pytorch_dump_folder_path,
            path_in_repo="transformer",
            repo_id="nielsr/absorbing-diffusion-churches",
            repo_type="model",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--checkpoint_path",
        default="/Users/nielsrogge/Documents/AbsorbingDiffusion/churches/absorbing_ema_2000000.th",
        type=str,
        help="Path to the original state dict (.th file).",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )
    parser.add_argument(
        "--push_to_hub", action="store_true", help="Whether or not to push the converted model to the 🤗 hub."
    )

    args = parser.parse_args()
    convert_transformer_checkpoint(args.checkpoint_path, args.pytorch_dump_folder_path, args.push_to_hub)
