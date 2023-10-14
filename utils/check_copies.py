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

"""
Copied verbatim from: https://github.com/huggingface/transformers/blob/main/utils/check_copies.py

Utility that checks whether the copies defined in the library match the original or not. This includes:
- All code commented with `# Copied from` comments,
- The list of models in the main README.md matches the ones in the localized READMEs and in the index.md,
- Files that are registered as full copies of one another in the `FULL_COPIES` constant of this script.

This also checks the list of models in the README is complete (has all models) and add a line to complete if there is
a model missing.

Use from the root of the repo with:

```bash
python utils/check_copies.py
```

for a check that will error in case of inconsistencies (used by `make repo-consistency`) or

```bash
python utils/check_copies.py --fix_and_overwrite
```

for a check that will fix all inconsistencies automatically (used by `make fix-copies`).
"""

import argparse
import glob
import os
import re

import black
from doc_builder.style_doc import style_docstrings_in_code


# All paths are set with the intent you should run this script from the root of the repo with the command
# python utils/check_copies.py
DIFFUSERS_PATH = "src/diffusers"
REPO_PATH = "."


def _should_continue(line, indent):
    # Helper function. Returns `True` if `line` is empty, starts with the `indent` or is the end parenthesis of a
    # function definition
    return line.startswith(indent) or len(line.strip()) == 0 or re.search(r"^\s*\)(\s*->.*:|:)\s*$", line) is not None


def find_code_in_diffusers(object_name):
    """Find and return the code source code of `object_name`."""
    parts = object_name.split(".")
    i = 0

    # First let's find the module where our object lives.
    module = parts[i]
    while i < len(parts) and not os.path.isfile(os.path.join(DIFFUSERS_PATH, f"{module}.py")):
        i += 1
        if i < len(parts):
            module = os.path.join(module, parts[i])
    if i >= len(parts):
        raise ValueError(f"`object_name` should begin with the name of a module of diffusers but got {object_name}.")

    with open(os.path.join(DIFFUSERS_PATH, f"{module}.py"), "r", encoding="utf-8", newline="\n") as f:
        lines = f.readlines()

    # Now let's find the class / func in the code!
    indent = ""
    line_index = 0
    for name in parts[i + 1 :]:
        while (
            line_index < len(lines) and re.search(rf"^{indent}(class|def)\s+{name}(\(|\:)", lines[line_index]) is None
        ):
            line_index += 1
        indent += "    "
        line_index += 1

    if line_index >= len(lines):
        raise ValueError(f" {object_name} does not match any function or class in {module}.")

    # We found the beginning of the class / func, now let's find the end (when the indent diminishes).
    start_index = line_index - 1
    while line_index < len(lines) and _should_continue(lines[line_index], indent):
        line_index += 1
    # Clean up empty lines at the end (if any).
    while len(lines[line_index - 1]) <= 1:
        line_index -= 1

    code_lines = lines[start_index:line_index]
    return "".join(code_lines)


_re_copy_warning = re.compile(r"^(\s*)#\s*Copied from\s+diffusers\.(\S+\.\S+)\s*($|\S.*$)")
_re_replace_pattern = re.compile(r"^\s*(\S+)->(\S+)(\s+.*|$)")
_re_fill_pattern = re.compile(r"<FILL\s+[^>]*>")


def get_indent(code):
    lines = code.split("\n")
    idx = 0
    while idx < len(lines) and len(lines[idx]) == 0:
        idx += 1
    if idx < len(lines):
        return re.search(r"^(\s*)\S", lines[idx]).groups()[0]
    return ""


def blackify(code):
    """
    Applies the black part of our `make style` command to `code`.
    """
    has_indent = len(get_indent(code)) > 0
    if has_indent:
        code = f"class Bla:\n{code}"
    mode = black.Mode(target_versions={black.TargetVersion.PY37}, line_length=119, preview=True)
    result = black.format_str(code, mode=mode)
    result, _ = style_docstrings_in_code(result)
    return result[len("class Bla:\n") :] if has_indent else result


def check_codes_match(observed_code, theoretical_code):
    """
    Checks if the code in `observed_code` and `theoretical_code` match with the exception of the class/function name.
    Returns the index of the first line where there is a difference (if any) and `None` if the codes match.
    """
    observed_code_header = observed_code.split("\n")[0]
    theoretical_code_header = theoretical_code.split("\n")[0]

    _re_class_match = re.compile(r"class\s+([^\(:]+)(?:\(|:)")
    _re_func_match = re.compile(r"def\s+([^\(]+)\(")
    for re_pattern in [_re_class_match, _re_func_match]:
        if re_pattern.match(observed_code_header) is not None:
            observed_obj_name = re_pattern.search(observed_code_header).groups()[0]
            theoretical_name = re_pattern.search(theoretical_code_header).groups()[0]
            theoretical_code_header = theoretical_code_header.replace(theoretical_name, observed_obj_name)

    diff_index = 0
    if theoretical_code_header != observed_code_header:
        return 0

    diff_index = 1
    for observed_line, theoretical_line in zip(observed_code.split("\n")[1:], theoretical_code.split("\n")[1:]):
        if observed_line != theoretical_line:
            return diff_index
        diff_index += 1


def is_copy_consistent(filename, overwrite=False):
    """
    Check if the code commented as a copy in `filename` matches the original.
    Return the differences or overwrites the content depending on `overwrite`.
    """
    with open(filename, "r", encoding="utf-8", newline="\n") as f:
        lines = f.readlines()
    diffs = []
    line_index = 0
    # Not a for loop cause `lines` is going to change (if `overwrite=True`).
    while line_index < len(lines):
        search = _re_copy_warning.search(lines[line_index])
        if search is None:
            line_index += 1
            continue

        # There is some copied code here, let's retrieve the original.
        indent, object_name, replace_pattern = search.groups()
        theoretical_code = find_code_in_diffusers(object_name)
        theoretical_indent = get_indent(theoretical_code)

        start_index = line_index + 1 if indent == theoretical_indent else line_index
        indent = theoretical_indent
        line_index = start_index + 1

        # Loop to check the observed code, stop when indentation diminishes or if we see a End copy comment.
        should_continue = True
        while line_index < len(lines) and should_continue:
            line_index += 1
            if line_index >= len(lines):
                break
            line = lines[line_index]
            # There is a special pattern `# End copy` to stop early. It's not documented cause it shouldn't really be
            # used.
            should_continue = _should_continue(line, indent) and re.search(f"^{indent}# End copy", line) is None
        # Clean up empty lines at the end (if any).
        while len(lines[line_index - 1]) <= 1:
            line_index -= 1

        observed_code_lines = lines[start_index:line_index]
        observed_code = "".join(observed_code_lines)

        # Remove any nested `Copied from` comments to avoid circular copies
        theoretical_code = [line for line in theoretical_code.split("\n") if _re_copy_warning.search(line) is None]
        theoretical_code = "\n".join(theoretical_code)

        # Before comparing, use the `replace_pattern` on the original code.
        if len(replace_pattern) > 0:
            patterns = replace_pattern.replace("with", "").split(",")
            patterns = [_re_replace_pattern.search(p) for p in patterns]
            for pattern in patterns:
                if pattern is None:
                    continue
                obj1, obj2, option = pattern.groups()
                theoretical_code = re.sub(obj1, obj2, theoretical_code)
                if option.strip() == "all-casing":
                    theoretical_code = re.sub(obj1.lower(), obj2.lower(), theoretical_code)
                    theoretical_code = re.sub(obj1.upper(), obj2.upper(), theoretical_code)

            theoretical_code = blackify(theoretical_code)

        # Test for a diff and act accordingly.
        diff_index = check_codes_match(observed_code, theoretical_code)
        if diff_index is not None:
            diffs.append([object_name, diff_index + start_index + 1])
            if overwrite:
                lines = lines[:start_index] + [theoretical_code] + lines[line_index:]
                line_index = start_index + 1

    if overwrite and len(diffs) > 0:
        # Warn the user a file has been modified.
        print(f"Detected changes, rewriting {filename}.")
        with open(filename, "w", encoding="utf-8", newline="\n") as f:
            f.writelines(lines)
    return diffs


def check_copies(overwrite: bool = False):
    """
    Check every file is copy-consistent with the original and maybe `overwrite` content when it is not. Also check the
    model list in the main README and other READMEs/index.md are consistent.
    """
    all_files = glob.glob(os.path.join(DIFFUSERS_PATH, "**/*.py"), recursive=True)
    diffs = []
    for filename in all_files:
        new_diffs = is_copy_consistent(filename, overwrite)
        diffs += [f"- {filename}: copy does not match {d[0]} at line {d[1]}" for d in new_diffs]
    if not overwrite and len(diffs) > 0:
        diff = "\n".join(diffs)
        raise Exception(
            "Found the following copy inconsistencies:\n"
            + diff
            + "\nRun `make fix-copies` or `python utils/check_copies.py --fix_and_overwrite` to fix them."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fix_and_overwrite", action="store_true", help="Whether to fix inconsistencies.")
    args = parser.parse_args()

    check_copies(args.fix_and_overwrite)
