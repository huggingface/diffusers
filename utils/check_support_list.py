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
Utility that checks that modules like attention processors are listed in the documentation file.

```bash
python utils/check_support_list.py
```

It has no auto-fix mode.
"""
import os
import re


# All paths are set with the intent you should run this script from the root of the repo with the command
# python utils/check_doctest_list.py
REPO_PATH = "."


def check_attention_processors():
    with open(os.path.join(REPO_PATH, "docs/source/en/api/attnprocessor.md"), "r") as f:
        doctext = f.read()
        matches = re.findall(r"\[\[autodoc\]\]\s([^\n]+)", doctext)
        documented_attention_processors = [match.split(".")[-1] for match in matches]

    with open(os.path.join(REPO_PATH, "src/diffusers/models/attention_processor.py"), "r") as f:
        doctext = f.read()
        processor_classes = re.findall(r"class\s+(\w+Processor(?:\d*_?\d*))[(:]", doctext)
        processor_classes = [proc for proc in processor_classes if "LoRA" not in proc and proc != "Attention"]

    for processor in processor_classes:
        if processor not in documented_attention_processors:
            raise ValueError(
                f"{processor} should be in listed in the attention processor documentation but is not. Please update the documentation."
            )

def check_image_processors():
    with open(os.path.join(REPO_PATH, "docs/source/en/api/image_processor.md"), "r") as f:
        doctext = f.read()
        matches = re.findall(r"\[\[autodoc\]\]\s([^\n]+)", doctext)
        documented_image_processors = [match.split(".")[-1] for match in matches]

    with open(os.path.join(REPO_PATH, "src/diffusers/image_processor.py"), "r") as f:
        doctext = f.read()
        processor_classes = re.findall(r"class\s+(\w+Processor(?:\d*_?\d*))[(:]", doctext)

    for processor in processor_classes:
        if processor not in documented_image_processors:
            raise ValueError(
                f"{processor} should be in listed in the image processor documentation but is not. Please update the documentation."
            )

def check_activations():
    with open(os.path.join(REPO_PATH, "docs/source/en/api/activations.md"), "r") as f:
        doctext = f.read()
        matches = re.findall(r"\[\[autodoc\]\]\s([^\n]+)", doctext)
        documented_activations = [match.split(".")[-1] for match in matches]

    with open(os.path.join(REPO_PATH, "src/diffusers/models/activations.py"), "r") as f:
        doctext = f.read()
        activation_classes = re.findall(r"class\s+(\w+)\s*\(.*?nn\.Module.*?\):", doctext)

    for activation in activation_classes:
        if activation not in documented_activations:
            raise ValueError(
                f"{activation} should be in listed in the activations documentation but is not. Please update the documentation."
            )


def check_normalizations():
    with open(os.path.join(REPO_PATH, "docs/source/en/api/normalization.md"), "r") as f:
        doctext = f.read()
        matches = re.findall(r"\[\[autodoc\]\]\s([^\n]+)", doctext)
        documented_normalizations = [match.split(".")[-1] for match in matches]

    with open(os.path.join(REPO_PATH, "src/diffusers/models/normalization.py"), "r") as f:
        doctext = f.read()
        normalization_classes = re.findall(r"class\s+(\w+)\s*\(.*?nn\.Module.*?\):", doctext)

    for norm in normalization_classes:
        if norm not in documented_normalizations:
            raise ValueError(
                f"{norm} should be in listed in the normalizations documentation but is not. Please update the documentation."
            )


if __name__ == "__main__":
    check_attention_processors()
    # check_image_processors()
    check_activations()
    check_normalizations()
