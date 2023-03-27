# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
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
from collections import defaultdict

import yaml


PATH_TO_TOC = "docs/source/en/_toctree.yml"


def clean_doc_toc(doc_list):
    """
    Cleans the table of content of the model documentation by removing duplicates and sorting models alphabetically.
    """
    counts = defaultdict(int)
    overview_doc = []
    new_doc_list = []
    for doc in doc_list:
        if "local" in doc:
            counts[doc["local"]] += 1

        if doc["title"].lower() == "overview":
            overview_doc.append({"local": doc["local"], "title": doc["title"]})
        else:
            new_doc_list.append(doc)

    doc_list = new_doc_list
    duplicates = [key for key, value in counts.items() if value > 1]

    new_doc = []
    for duplicate_key in duplicates:
        titles = list({doc["title"] for doc in doc_list if doc["local"] == duplicate_key})
        if len(titles) > 1:
            raise ValueError(
                f"{duplicate_key} is present several times in the documentation table of content at "
                "`docs/source/en/_toctree.yml` with different *Title* values. Choose one of those and remove the "
                "others."
            )
        # Only add this once
        new_doc.append({"local": duplicate_key, "title": titles[0]})

    # Add none duplicate-keys
    new_doc.extend([doc for doc in doc_list if "local" not in counts or counts[doc["local"]] == 1])
    new_doc = sorted(new_doc, key=lambda s: s["title"].lower())

    # "overview" gets special treatment and is always first
    if len(overview_doc) > 1:
        raise ValueError("{doc_list} has two 'overview' docs which is not allowed.")

    overview_doc.extend(new_doc)

    # Sort
    return overview_doc


def check_scheduler_doc(overwrite=False):
    with open(PATH_TO_TOC, encoding="utf-8") as f:
        content = yaml.safe_load(f.read())

    # Get to the API doc
    api_idx = 0
    while content[api_idx]["title"] != "API":
        api_idx += 1
    api_doc = content[api_idx]["sections"]

    # Then to the model doc
    scheduler_idx = 0
    while api_doc[scheduler_idx]["title"] != "Schedulers":
        scheduler_idx += 1

    scheduler_doc = api_doc[scheduler_idx]["sections"]
    new_scheduler_doc = clean_doc_toc(scheduler_doc)

    diff = False
    if new_scheduler_doc != scheduler_doc:
        diff = True
        if overwrite:
            api_doc[scheduler_idx]["sections"] = new_scheduler_doc

    if diff:
        if overwrite:
            content[api_idx]["sections"] = api_doc
            with open(PATH_TO_TOC, "w", encoding="utf-8") as f:
                f.write(yaml.dump(content, allow_unicode=True))
        else:
            raise ValueError(
                "The model doc part of the table of content is not properly sorted, run `make style` to fix this."
            )


def check_pipeline_doc(overwrite=False):
    with open(PATH_TO_TOC, encoding="utf-8") as f:
        content = yaml.safe_load(f.read())

    # Get to the API doc
    api_idx = 0
    while content[api_idx]["title"] != "API":
        api_idx += 1
    api_doc = content[api_idx]["sections"]

    # Then to the model doc
    pipeline_idx = 0
    while api_doc[pipeline_idx]["title"] != "Pipelines":
        pipeline_idx += 1

    diff = False
    pipeline_docs = api_doc[pipeline_idx]["sections"]
    new_pipeline_docs = []

    # sort sub pipeline docs
    for pipeline_doc in pipeline_docs:
        if "section" in pipeline_doc:
            sub_pipeline_doc = pipeline_doc["section"]
            new_sub_pipeline_doc = clean_doc_toc(sub_pipeline_doc)
            if overwrite:
                pipeline_doc["section"] = new_sub_pipeline_doc
        new_pipeline_docs.append(pipeline_doc)

    # sort overall pipeline doc
    new_pipeline_docs = clean_doc_toc(new_pipeline_docs)

    if new_pipeline_docs != pipeline_docs:
        diff = True
        if overwrite:
            api_doc[pipeline_idx]["sections"] = new_pipeline_docs

    if diff:
        if overwrite:
            content[api_idx]["sections"] = api_doc
            with open(PATH_TO_TOC, "w", encoding="utf-8") as f:
                f.write(yaml.dump(content, allow_unicode=True))
        else:
            raise ValueError(
                "The model doc part of the table of content is not properly sorted, run `make style` to fix this."
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fix_and_overwrite", action="store_true", help="Whether to fix inconsistencies.")
    args = parser.parse_args()

    check_scheduler_doc(args.fix_and_overwrite)
    check_pipeline_doc(args.fix_and_overwrite)
