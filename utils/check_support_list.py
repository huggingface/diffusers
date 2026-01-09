# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
"""
Utility that checks that modules like attention processors are listed in the documentation file.

```bash
python utils/check_support_list.py
```

It has no auto-fix mode.
"""

import os
import re


# All paths are set with the intent that you run this script from the root of the repo
REPO_PATH = "."


def read_documented_classes(doc_path, autodoc_regex=r"\[\[autodoc\]\]\s([^\n]+)"):
    """
    Reads documented classes from a doc file using a regex to find lines like [[autodoc]] my.module.Class.
    Returns a list of documented class names (just the class name portion).
    """
    with open(os.path.join(REPO_PATH, doc_path), "r") as f:
        doctext = f.read()
    matches = re.findall(autodoc_regex, doctext)
    return [match.split(".")[-1] for match in matches]


def read_source_classes(src_path, class_regex, exclude_conditions=None):
    """
    Reads class names from a source file using a regex that captures class definitions.
    Optionally exclude classes based on a list of conditions (functions that take class name and return bool).
    """
    if exclude_conditions is None:
        exclude_conditions = []
    with open(os.path.join(REPO_PATH, src_path), "r") as f:
        doctext = f.read()
    classes = re.findall(class_regex, doctext)
    # Filter out classes that meet any of the exclude conditions
    filtered_classes = [c for c in classes if not any(cond(c) for cond in exclude_conditions)]
    return filtered_classes


def check_documentation(doc_path, src_path, doc_regex, src_regex, exclude_conditions=None):
    """
    Generic function to check if all classes defined in `src_path` are documented in `doc_path`.
    Returns a set of undocumented class names.
    """
    documented = set(read_documented_classes(doc_path, doc_regex))
    source_classes = set(read_source_classes(src_path, src_regex, exclude_conditions=exclude_conditions))

    # Find which classes in source are not documented in a deterministic way.
    undocumented = sorted(source_classes - documented)
    return undocumented


if __name__ == "__main__":
    # Define the checks we need to perform
    checks = {
        "Attention Processors": {
            "doc_path": "docs/source/en/api/attnprocessor.md",
            "src_path": "src/diffusers/models/attention_processor.py",
            "doc_regex": r"\[\[autodoc\]\]\s([^\n]+)",
            "src_regex": r"class\s+(\w+Processor(?:\d*_?\d*))[:(]",
            "exclude_conditions": [lambda c: "LoRA" in c, lambda c: c == "Attention"],
        },
        "Image Processors": {
            "doc_path": "docs/source/en/api/image_processor.md",
            "src_path": "src/diffusers/image_processor.py",
            "doc_regex": r"\[\[autodoc\]\]\s([^\n]+)",
            "src_regex": r"class\s+(\w+Processor(?:\d*_?\d*))[:(]",
        },
        "Activations": {
            "doc_path": "docs/source/en/api/activations.md",
            "src_path": "src/diffusers/models/activations.py",
            "doc_regex": r"\[\[autodoc\]\]\s([^\n]+)",
            "src_regex": r"class\s+(\w+)\s*\(.*?nn\.Module.*?\):",
        },
        "Normalizations": {
            "doc_path": "docs/source/en/api/normalization.md",
            "src_path": "src/diffusers/models/normalization.py",
            "doc_regex": r"\[\[autodoc\]\]\s([^\n]+)",
            "src_regex": r"class\s+(\w+)\s*\(.*?nn\.Module.*?\):",
            "exclude_conditions": [
                # Exclude LayerNorm as it's an intentional exception
                lambda c: c == "LayerNorm"
            ],
        },
        "LoRA Mixins": {
            "doc_path": "docs/source/en/api/loaders/lora.md",
            "src_path": "src/diffusers/loaders/lora_pipeline.py",
            "doc_regex": r"\[\[autodoc\]\]\s([^\n]+)",
            "src_regex": r"class\s+(\w+LoraLoaderMixin(?:\d*_?\d*))[:(]",
        },
    }

    missing_items = {}
    for category, params in checks.items():
        undocumented = check_documentation(
            doc_path=params["doc_path"],
            src_path=params["src_path"],
            doc_regex=params["doc_regex"],
            src_regex=params["src_regex"],
            exclude_conditions=params.get("exclude_conditions"),
        )
        if undocumented:
            missing_items[category] = undocumented

    # If we have any missing items, raise a single combined error
    if missing_items:
        error_msg = ["Some classes are not documented properly:\n"]
        for category, classes in missing_items.items():
            error_msg.append(f"- {category}: {', '.join(sorted(classes))}")
        raise ValueError("\n".join(error_msg))
