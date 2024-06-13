# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
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
"""
Utility that updates the metadata of the Diffusers library in the repository `huggingface/diffusers-metadata`.

Usage for an update (as used by the GitHub action `update_metadata`):

```bash
python utils/update_metadata.py
```

Script modified from:
https://github.com/huggingface/transformers/blob/main/utils/update_metadata.py
"""

import argparse
import os
import tempfile

import pandas as pd
from datasets import Dataset
from huggingface_hub import hf_hub_download, upload_folder

from diffusers.pipelines.auto_pipeline import (
    AUTO_IMAGE2IMAGE_PIPELINES_MAPPING,
    AUTO_INPAINT_PIPELINES_MAPPING,
    AUTO_TEXT2IMAGE_PIPELINES_MAPPING,
)


PIPELINE_TAG_JSON = "pipeline_tags.json"


def get_supported_pipeline_table() -> dict:
    """
    Generates a dictionary containing the supported auto classes for each pipeline type,
    using the content of the auto modules.
    """
    # All supported pipelines for automatic mapping.
    all_supported_pipeline_classes = [
        (class_name.__name__, "text-to-image", "AutoPipelineForText2Image")
        for _, class_name in AUTO_TEXT2IMAGE_PIPELINES_MAPPING.items()
    ]
    all_supported_pipeline_classes += [
        (class_name.__name__, "image-to-image", "AutoPipelineForImage2Image")
        for _, class_name in AUTO_IMAGE2IMAGE_PIPELINES_MAPPING.items()
    ]
    all_supported_pipeline_classes += [
        (class_name.__name__, "image-to-image", "AutoPipelineForInpainting")
        for _, class_name in AUTO_INPAINT_PIPELINES_MAPPING.items()
    ]
    all_supported_pipeline_classes = list(set(all_supported_pipeline_classes))
    all_supported_pipeline_classes.sort(key=lambda x: x[0])

    data = {}
    data["pipeline_class"] = [sample[0] for sample in all_supported_pipeline_classes]
    data["pipeline_tag"] = [sample[1] for sample in all_supported_pipeline_classes]
    data["auto_class"] = [sample[2] for sample in all_supported_pipeline_classes]

    return data


def update_metadata(commit_sha: str):
    """
    Update the metadata for the Diffusers repo in `huggingface/diffusers-metadata`.

    Args:
        commit_sha (`str`): The commit SHA on Diffusers corresponding to this update.
    """
    pipelines_table = get_supported_pipeline_table()
    pipelines_table = pd.DataFrame(pipelines_table)
    pipelines_dataset = Dataset.from_pandas(pipelines_table)

    hub_pipeline_tags_json = hf_hub_download(
        repo_id="huggingface/diffusers-metadata",
        filename=PIPELINE_TAG_JSON,
        repo_type="dataset",
    )
    with open(hub_pipeline_tags_json) as f:
        hub_pipeline_tags_json = f.read()

    with tempfile.TemporaryDirectory() as tmp_dir:
        pipelines_dataset.to_json(os.path.join(tmp_dir, PIPELINE_TAG_JSON))

        with open(os.path.join(tmp_dir, PIPELINE_TAG_JSON)) as f:
            pipeline_tags_json = f.read()

        hub_pipeline_tags_equal = hub_pipeline_tags_json == pipeline_tags_json
        if hub_pipeline_tags_equal:
            print("No updates, not pushing the metadata files.")
            return

        if commit_sha is not None:
            commit_message = (
                f"Update with commit {commit_sha}\n\nSee: "
                f"https://github.com/huggingface/diffusers/commit/{commit_sha}"
            )
        else:
            commit_message = "Update"

        upload_folder(
            repo_id="huggingface/diffusers-metadata",
            folder_path=tmp_dir,
            repo_type="dataset",
            commit_message=commit_message,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit_sha", default=None, type=str, help="The sha of the commit going with this update.")
    args = parser.parse_args()

    update_metadata(args.commit_sha)
