#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

import argparse

import pandas as pd
import torch
from datasets import load_dataset
from huggingface_hub.utils import insecure_hashlib
from tqdm.auto import tqdm
from transformers import T5EncoderModel

from diffusers import FluxPipeline


MAX_SEQ_LENGTH = 77
OUTPUT_PATH = "embeddings.parquet"


def generate_image_hash(image):
    return insecure_hashlib.sha256(image.tobytes()).hexdigest()


def load_flux_dev_pipeline():
    id = "black-forest-labs/FLUX.1-dev"
    text_encoder = T5EncoderModel.from_pretrained(id, subfolder="text_encoder_2", load_in_8bit=True, device_map="auto")
    pipeline = FluxPipeline.from_pretrained(
        id, text_encoder_2=text_encoder, transformer=None, vae=None, device_map="balanced"
    )
    return pipeline


@torch.no_grad()
def compute_embeddings(pipeline, prompts, max_sequence_length):
    all_prompt_embeds = []
    all_pooled_prompt_embeds = []
    all_text_ids = []
    for prompt in tqdm(prompts, desc="Encoding prompts."):
        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = pipeline.encode_prompt(prompt=prompt, prompt_2=None, max_sequence_length=max_sequence_length)
        all_prompt_embeds.append(prompt_embeds)
        all_pooled_prompt_embeds.append(pooled_prompt_embeds)
        all_text_ids.append(text_ids)

    max_memory = torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024
    print(f"Max memory allocated: {max_memory:.3f} GB")
    return all_prompt_embeds, all_pooled_prompt_embeds, all_text_ids


def run(args):
    dataset = load_dataset("Norod78/Yarn-art-style", split="train")
    image_prompts = {generate_image_hash(sample["image"]): sample["text"] for sample in dataset}
    all_prompts = list(image_prompts.values())
    print(f"{len(all_prompts)=}")

    pipeline = load_flux_dev_pipeline()
    all_prompt_embeds, all_pooled_prompt_embeds, all_text_ids = compute_embeddings(
        pipeline, all_prompts, args.max_sequence_length
    )

    data = []
    for i, (image_hash, _) in enumerate(image_prompts.items()):
        data.append((image_hash, all_prompt_embeds[i], all_pooled_prompt_embeds[i], all_text_ids[i]))
    print(f"{len(data)=}")

    # Create a DataFrame
    embedding_cols = ["prompt_embeds", "pooled_prompt_embeds", "text_ids"]
    df = pd.DataFrame(data, columns=["image_hash"] + embedding_cols)
    print(f"{len(df)=}")

    # Convert embedding lists to arrays (for proper storage in parquet)
    for col in embedding_cols:
        df[col] = df[col].apply(lambda x: x.cpu().numpy().flatten().tolist())

    # Save the dataframe to a parquet file
    df.to_parquet(args.output_path)
    print(f"Data successfully serialized to {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=MAX_SEQ_LENGTH,
        help="Maximum sequence length to use for computing the embeddings. The more the higher computational costs.",
    )
    parser.add_argument("--output_path", type=str, default=OUTPUT_PATH, help="Path to serialize the parquet file.")
    args = parser.parse_args()

    run(args)
