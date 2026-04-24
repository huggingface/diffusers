# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os

from tqdm import tqdm

"""example command
python -m scripts.create_prompts_for_gr1_dataset --dataset_path datasets/benchmark_train/gr1
"""


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create text prompts for GR1 dataset")
    parser.add_argument(
        "--dataset_path", type=str, default="datasets/benchmark_train/gr1", help="Root path to the dataset"
    )
    parser.add_argument(
        "--prompt_prefix", type=str, default="The robot arm is performing a task. ", help="Prefix of the prompt"
    )
    parser.add_argument(
        "--meta_csv", type=str, default="datasets/benchmark_train/gr1/metadata.csv", help="Metadata csv file"
    )
    return parser.parse_args()


def main(args) -> None:
    meta_csv = args.meta_csv
    meta_lines = open(meta_csv).readlines()[1:]
    meta_txt_dir = os.path.join(args.dataset_path, "metas")
    os.makedirs(meta_txt_dir, exist_ok=True)

    for meta_line in tqdm(meta_lines):
        video_filename, prompt = meta_line.split(",", 1)
        prompt = prompt.strip("\n")
        if prompt.startswith('"') and prompt.endswith('"'):
            # Remove the quotes
            prompt = prompt[1:-1]
        prompt = args.prompt_prefix + prompt
        meta_txt_filename = os.path.join(meta_txt_dir, os.path.basename(video_filename).replace(".mp4", ".txt"))
        with open(meta_txt_filename, "w") as fp:
            fp.write(prompt)

        print(f"encoding prompt: {prompt}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
