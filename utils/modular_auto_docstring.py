# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team.
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
Auto Docstring Generator for Modular Pipeline Blocks

This script scans Python files for classes that have `# auto_docstring` comment above them
and inserts/updates the docstring from the class's `doc` property.

Run from the root of the repo:
    python utils/modular_auto_docstring.py [path] [--fix_and_overwrite]

Examples:
    # Check for auto_docstring markers (will error if found without proper docstring)
    python utils/modular_auto_docstring.py

    # Check specific directory
    python utils/modular_auto_docstring.py src/diffusers/modular_pipelines/

    # Fix and overwrite the docstrings
    python utils/modular_auto_docstring.py --fix_and_overwrite

Usage in code:
    # auto_docstring
    class QwenImageAutoVaeEncoderStep(AutoPipelineBlocks):
        # docstring will be automatically inserted here

        @property
        def doc(self):
            return "Your docstring content..."
"""

import argparse
import ast
import glob
import importlib
import os
import re
import subprocess
import sys


# All paths are set with the intent you should run this script from the root of the repo
DIFFUSERS_PATH = "src/diffusers"
REPO_PATH = "."

# Pattern to match the auto_docstring comment
AUTO_DOCSTRING_PATTERN = re.compile(r"^\s*#\s*auto_docstring\s*$")


def setup_diffusers_import():
    """Setup import path to use the local diffusers module."""
    src_path = os.path.join(REPO_PATH, "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


def get_module_from_filepath(filepath: str) -> str:
    """Convert a filepath to a module name."""
    filepath = os.path.normpath(filepath)

    if filepath.startswith("src" + os.sep):
        filepath = filepath[4:]

    if filepath.endswith(".py"):
        filepath = filepath[:-3]

    module_name = filepath.replace(os.sep, ".")
    return module_name


def load_module(filepath: str):
    """Load a module from filepath."""
    setup_diffusers_import()
    module_name = get_module_from_filepath(filepath)

    try:
        module = importlib.import_module(module_name)
        return module
    except Exception as e:
        print(f"Warning: Could not import module {module_name}: {e}")
        return None


def get_doc_from_class(module, class_name: str) -> str:
    """Get the doc property from an instantiated class."""
    if module is None:
        return None

    cls = getattr(module, class_name, None)
    if cls is None:
        return None

    try:
        instance = cls()
        if hasattr(instance, "doc"):
            return instance.doc
    except Exception as e:
        print(f"Warning: Could not instantiate {class_name}: {e}")

    return None


def find_auto_docstring_classes(filepath: str) -> list:
    """
    Find all classes in a file that have # auto_docstring comment above them.

    Returns list of (class_name, class_line_number, has_existing_docstring, docstring_end_line)
    """
    with open(filepath, "r", encoding="utf-8", newline="\n") as f:
        lines = f.readlines()

    # Parse AST to find class locations and their docstrings
    content = "".join(lines)
    try:
        tree = ast.parse(content)
    except SyntaxError as e:
        print(f"Syntax error in {filepath}: {e}")
        return []

    # Build a map of class_name -> (class_line, has_docstring, docstring_end_line)
    class_info = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            has_docstring = False
            docstring_end_line = node.lineno  # default to class line

            if node.body and isinstance(node.body[0], ast.Expr):
                first_stmt = node.body[0]
                if isinstance(first_stmt.value, ast.Constant) and isinstance(first_stmt.value.value, str):
                    has_docstring = True
                    docstring_end_line = first_stmt.end_lineno or first_stmt.lineno

            class_info[node.name] = (node.lineno, has_docstring, docstring_end_line)

    # Now scan for # auto_docstring comments
    classes_to_update = []

    for i, line in enumerate(lines):
        if AUTO_DOCSTRING_PATTERN.match(line):
            # Found the marker, look for class definition on next non-empty, non-comment line
            j = i + 1
            while j < len(lines):
                next_line = lines[j].strip()
                if next_line and not next_line.startswith("#"):
                    break
                j += 1

            if j < len(lines) and lines[j].strip().startswith("class "):
                # Extract class name
                match = re.match(r"class\s+(\w+)", lines[j].strip())
                if match:
                    class_name = match.group(1)
                    if class_name in class_info:
                        class_line, has_docstring, docstring_end_line = class_info[class_name]
                        classes_to_update.append((class_name, class_line, has_docstring, docstring_end_line))

    return classes_to_update


def strip_class_name_line(doc: str, class_name: str) -> str:
    """Remove the 'class ClassName' line from the doc if present."""
    lines = doc.strip().split("\n")
    if lines and lines[0].strip() == f"class {class_name}":
        # Remove the class line and any blank line following it
        lines = lines[1:]
        while lines and not lines[0].strip():
            lines = lines[1:]
    return "\n".join(lines)


def format_docstring(doc: str, indent: str = "    ") -> str:
    """Format a doc string as a properly indented docstring."""
    lines = doc.strip().split("\n")

    if len(lines) == 1:
        return f'{indent}"""{lines[0]}"""\n'
    else:
        result = [f'{indent}"""\n']
        for line in lines:
            if line.strip():
                result.append(f"{indent}{line}\n")
            else:
                result.append("\n")
        result.append(f'{indent}"""\n')
        return "".join(result)


def run_ruff_format(filepath: str):
    """Run ruff check --fix, ruff format, and doc-builder style on a file to ensure consistent formatting."""
    try:
        # First run ruff check --fix to fix any linting issues (including line length)
        subprocess.run(
            ["ruff", "check", "--fix", filepath],
            check=False,  # Don't fail if there are unfixable issues
            capture_output=True,
            text=True,
        )
        # Then run ruff format for code formatting
        subprocess.run(
            ["ruff", "format", filepath],
            check=True,
            capture_output=True,
            text=True,
        )
        # Finally run doc-builder style for docstring formatting
        subprocess.run(
            ["doc-builder", "style", filepath, "--max_len", "119"],
            check=False,  # Don't fail if doc-builder has issues
            capture_output=True,
            text=True,
        )
        print(f"Formatted {filepath}")
    except subprocess.CalledProcessError as e:
        print(f"Warning: formatting failed for {filepath}: {e.stderr}")
    except FileNotFoundError as e:
        print(f"Warning: tool not found ({e}). Skipping formatting.")
    except Exception as e:
        print(f"Warning: unexpected error formatting {filepath}: {e}")


def get_existing_docstring(lines: list, class_line: int, docstring_end_line: int) -> str:
    """Extract the existing docstring content from lines."""
    # class_line is 1-indexed, docstring starts at class_line (0-indexed: class_line)
    # and ends at docstring_end_line (1-indexed, inclusive)
    docstring_lines = lines[class_line:docstring_end_line]
    return "".join(docstring_lines)


def process_file(filepath: str, overwrite: bool = False) -> list:
    """
    Process a file and find/insert docstrings for # auto_docstring marked classes.

    Returns list of classes that need updating.
    """
    classes_to_update = find_auto_docstring_classes(filepath)

    if not classes_to_update:
        return []

    if not overwrite:
        # Check mode: only verify that docstrings exist
        # Content comparison is not reliable due to formatting differences
        classes_needing_update = []
        for class_name, class_line, has_docstring, docstring_end_line in classes_to_update:
            if not has_docstring:
                # No docstring exists, needs update
                classes_needing_update.append((filepath, class_name, class_line))
        return classes_needing_update

    # Load the module to get doc properties
    module = load_module(filepath)

    with open(filepath, "r", encoding="utf-8", newline="\n") as f:
        lines = f.readlines()

    # Process in reverse order to maintain line numbers
    updated = False
    for class_name, class_line, has_docstring, docstring_end_line in reversed(classes_to_update):
        doc = get_doc_from_class(module, class_name)

        if doc is None:
            print(f"Warning: Could not get doc for {class_name} in {filepath}")
            continue

        # Remove the "class ClassName" line since it's redundant in a docstring
        doc = strip_class_name_line(doc, class_name)

        # Format the new docstring with 4-space indent
        new_docstring = format_docstring(doc, "    ")

        if has_docstring:
            # Replace existing docstring (line after class definition to docstring_end_line)
            # class_line is 1-indexed, we want to replace from class_line+1 to docstring_end_line
            lines = lines[:class_line] + [new_docstring] + lines[docstring_end_line:]
        else:
            # Insert new docstring right after class definition line
            # class_line is 1-indexed, so lines[class_line-1] is the class line
            # Insert at position class_line (which is right after the class line)
            lines = lines[:class_line] + [new_docstring] + lines[class_line:]

        updated = True
        print(f"Updated docstring for {class_name} in {filepath}")

    if updated:
        with open(filepath, "w", encoding="utf-8", newline="\n") as f:
            f.writelines(lines)
        # Run ruff format to ensure consistent line wrapping
        run_ruff_format(filepath)

    return [(filepath, cls_name, line) for cls_name, line, _, _ in classes_to_update]


def check_auto_docstrings(path: str = None, overwrite: bool = False):
    """
    Check all files for # auto_docstring markers and optionally fix them.
    """
    if path is None:
        path = DIFFUSERS_PATH

    if os.path.isfile(path):
        all_files = [path]
    else:
        all_files = glob.glob(os.path.join(path, "**/*.py"), recursive=True)

    all_markers = []

    for filepath in all_files:
        markers = process_file(filepath, overwrite)
        all_markers.extend(markers)

    if not overwrite and len(all_markers) > 0:
        message = "\n".join([f"- {f}: {cls} at line {line}" for f, cls, line in all_markers])
        raise ValueError(
            f"Found the following # auto_docstring markers that need docstrings:\n{message}\n\n"
            f"Run `python utils/modular_auto_docstring.py --fix_and_overwrite` to fix them."
        )

    if overwrite and len(all_markers) > 0:
        print(f"\nProcessed {len(all_markers)} docstring(s).")
    elif not overwrite and len(all_markers) == 0:
        print("All # auto_docstring markers have valid docstrings.")
    elif len(all_markers) == 0:
        print("No # auto_docstring markers found.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check and fix # auto_docstring markers in modular pipeline blocks",
    )
    parser.add_argument("path", nargs="?", default=None, help="File or directory to process (default: src/diffusers)")
    parser.add_argument(
        "--fix_and_overwrite",
        action="store_true",
        help="Whether to fix the docstrings by inserting them from doc property.",
    )

    args = parser.parse_args()

    check_auto_docstrings(args.path, args.fix_and_overwrite)
