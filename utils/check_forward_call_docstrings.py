# coding=utf-8
# Copyright 2026 The HuggingFace Inc. team.
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
Check that arguments of ``forward()`` (for models) and ``__call__()`` (for
pipelines) match the method's docstring exactly:

* every signature argument has an entry in the ``Args:`` /
  ``Arguments:`` / ``Parameters:`` section, and
* every documented argument still exists in the signature
  (stale entries from removed/renamed args are flagged).

A "main" class is detected via its base classes — models inherit from
``ModelMixin`` and pipelines inherit from ``DiffusionPipeline``. Only methods
defined directly on the class are checked; inherited methods are checked when
the parent class is visited.

Run from the repository root:

    python utils/check_forward_call_docstrings.py

Optionally restrict to specific files:

    python utils/check_forward_call_docstrings.py --paths src/diffusers/models/transformers/transformer_flux.py

Auto-fix stale (documented-but-removed) entries — missing entries are never
auto-added (no placeholders), only stale ones are removed:

    python utils/check_forward_call_docstrings.py --fix
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "src" / "diffusers" / "models"
PIPELINES_DIR = REPO_ROOT / "src" / "diffusers" / "pipelines"

MODEL_BASE = "ModelMixin"
PIPELINE_BASE = "DiffusionPipeline"

SECTION_HEADERS = {
    "Args:",
    "Arguments:",
    "Parameters:",
    "Returns:",
    "Return:",
    "Yields:",
    "Raises:",
    "Examples:",
    "Example:",
    "Note:",
    "Notes:",
    "References:",
    "See Also:",
}

# `name (...)` or `name:` at the start of a (stripped) line.
_ARG_HEADER_RE = re.compile(r"^([A-Za-z_]\w*)\s*[(:]")

# Pairs of (class_name, method_name) whose missing-arg errors should be
# suppressed. Use sparingly — prefer fixing the docstring.
IGNORE: set[tuple[str, str]] = set()


def _base_class_names(class_def: ast.ClassDef) -> set[str]:
    """Return the textual names of base classes (best-effort)."""
    names: set[str] = set()
    for base in class_def.bases:
        if isinstance(base, ast.Name):
            names.add(base.id)
        elif isinstance(base, ast.Attribute):
            names.add(base.attr)
    return names


def _find_method(class_def: ast.ClassDef, method_name: str) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    for node in class_def.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == method_name:
            return node
    return None


def _signature_arg_names(func: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    args = func.args
    collected: list[str] = []
    for a in (*args.posonlyargs, *args.args, *args.kwonlyargs):
        if a.arg == "self" or a.arg == "cls":
            continue
        collected.append(a.arg)
    return collected


def _extract_documented_args(docstring: str | None) -> set[str]:
    """Extract argument names listed in an Args/Arguments/Parameters section.

    Assumes the docstring has been cleaned (``inspect.cleandoc`` / ``ast.get_docstring``).
    The section ends at the next blank-line-followed-by-section-header or at the
    end of the docstring.
    """
    if not docstring:
        return set()

    lines = docstring.splitlines()

    # Locate the Args/Arguments/Parameters header.
    start = None
    header_indent = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped in {"Args:", "Arguments:", "Parameters:"}:
            start = i + 1
            header_indent = len(line) - len(line.lstrip())
            break
    if start is None:
        return set()

    # First non-empty line after the header sets the per-entry indent level.
    entry_indent: int | None = None
    documented: set[str] = set()

    for line in lines[start:]:
        stripped = line.strip()
        if not stripped:
            continue
        indent = len(line) - len(line.lstrip())

        # A new section at the same (or shallower) indent ends the args block.
        if indent <= header_indent and stripped in SECTION_HEADERS:
            break

        if entry_indent is None:
            entry_indent = indent

        # Only lines at the entry indent are candidate arg headers; deeper
        # indents are descriptions/continuations.
        if indent != entry_indent:
            continue

        match = _ARG_HEADER_RE.match(stripped)
        if match:
            documented.add(match.group(1))

    return documented


def check_file(path: Path, kind: str) -> list[str]:
    """Return a list of human-readable error strings for ``path``."""
    method_name = "forward" if kind == "model" else "__call__"
    base_class = MODEL_BASE if kind == "model" else PIPELINE_BASE

    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return []

    errors: list[str] = []
    rel = path.relative_to(REPO_ROOT)

    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        if base_class not in _base_class_names(node):
            continue
        if (node.name, method_name) in IGNORE:
            continue
        method = _find_method(node, method_name)
        if method is None:
            continue
        sig_args = _signature_arg_names(method)
        if not sig_args:
            continue
        sig_set = set(sig_args)
        documented = _extract_documented_args(ast.get_docstring(method))
        missing = [a for a in sig_args if a not in documented]
        stale = sorted(documented - sig_set)
        if missing:
            errors.append(
                f"{rel}:{method.lineno}: {node.name}.{method_name} is missing "
                f"docstring entries for: {', '.join(missing)}"
            )
        if stale:
            errors.append(
                f"{rel}:{method.lineno}: {node.name}.{method_name} documents "
                f"argument(s) not in the signature: {', '.join(stale)}"
            )
    return errors


def fix_file(path: Path, kind: str) -> list[str]:
    """Remove stale arg entries (documented but not in signature) in-place.

    Missing-in-signature → docstring entries are NOT added (no placeholders).
    Returns a list of ``"ClassName.method: removed name1, name2"`` strings
    describing what was removed.
    """
    method_name = "forward" if kind == "model" else "__call__"
    base_class = MODEL_BASE if kind == "model" else PIPELINE_BASE

    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source)
    except (SyntaxError, UnicodeDecodeError):
        return []

    lines = source.splitlines(keepends=True)
    # (start_idx, end_idx_exclusive) ranges of lines to drop.
    deletions: list[tuple[int, int]] = []
    summaries: list[str] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        if base_class not in _base_class_names(node):
            continue
        method = _find_method(node, method_name)
        if method is None:
            continue
        # Method must start with a string docstring expression.
        if not (
            method.body
            and isinstance(method.body[0], ast.Expr)
            and isinstance(method.body[0].value, ast.Constant)
            and isinstance(method.body[0].value.value, str)
        ):
            continue

        sig_set = set(_signature_arg_names(method))
        documented = _extract_documented_args(ast.get_docstring(method))
        stale = documented - sig_set
        if not stale:
            continue

        docstring_expr = method.body[0]
        doc_start = docstring_expr.lineno - 1  # 0-indexed
        doc_end = docstring_expr.end_lineno - 1  # 0-indexed, inclusive

        # Locate the Args/Arguments/Parameters header in raw source.
        args_idx: int | None = None
        header_indent = 0
        for i in range(doc_start, doc_end + 1):
            stripped = lines[i].strip()
            if stripped in {"Args:", "Arguments:", "Parameters:"}:
                args_idx = i
                header_indent = len(lines[i]) - len(lines[i].lstrip())
                break
        if args_idx is None:
            continue

        # First non-empty line after the header sets the per-entry indent.
        entry_indent: int | None = None
        for i in range(args_idx + 1, doc_end + 1):
            stripped = lines[i].strip()
            if not stripped:
                continue
            entry_indent = len(lines[i]) - len(lines[i].lstrip())
            break
        if entry_indent is None or entry_indent <= header_indent:
            continue

        # Walk entries; each entry spans from its header line up to (but not
        # including) the next entry header / section header / end of docstring.
        current_name: str | None = None
        current_start: int = -1
        end_of_args: int | None = None

        for i in range(args_idx + 1, doc_end + 1):
            line = lines[i]
            stripped = line.strip()
            if not stripped:
                continue
            indent = len(line) - len(line.lstrip())

            if indent <= header_indent and stripped in SECTION_HEADERS:
                end_of_args = i
                break

            if indent == entry_indent:
                m = _ARG_HEADER_RE.match(stripped)
                if m:
                    if current_name in stale:
                        deletions.append((current_start, i))
                    current_name = m.group(1)
                    current_start = i

        if current_name in stale:
            end = end_of_args if end_of_args is not None else doc_end
            # Trailing blank lines belong to inter-section spacing (or the
            # blank line before the closing """), not to this entry.
            while end > current_start + 1 and not lines[end - 1].strip():
                end -= 1
            deletions.append((current_start, end))

        summaries.append(f"{node.name}.{method_name}: removed {', '.join(sorted(stale))}")

    if not deletions:
        return []

    deletions.sort()
    new_lines = list(lines)
    for start, end in reversed(deletions):
        del new_lines[start:end]
    path.write_text("".join(new_lines), encoding="utf-8")
    return summaries


def _kind_for_path(path: Path) -> str | None:
    parts = path.resolve().parts
    if "pipelines" in parts:
        return "pipeline"
    if "models" in parts:
        return "model"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paths",
        nargs="+",
        help="Specific files to check (defaults to all of src/diffusers/{models,pipelines}).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Debug helper: when --paths is not given, only check the first N files "
            "(in sorted order) from each of models/ and pipelines/."
        ),
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help=(
            "Remove stale (documented-but-not-in-signature) argument entries from "
            "docstrings in-place. Missing-in-docstring entries are NOT auto-added "
            "(no placeholders) and will still be reported."
        ),
    )
    args = parser.parse_args()

    targets: list[tuple[Path, str]] = []
    if args.paths:
        for raw in args.paths:
            p = Path(raw).resolve()
            kind = _kind_for_path(p)
            if kind is None:
                print(f"Skipping {raw}: not under models/ or pipelines/.", file=sys.stderr)
                continue
            targets.append((p, kind))
    else:
        model_files = sorted(MODELS_DIR.rglob("*.py"))
        pipeline_files = sorted(PIPELINES_DIR.rglob("*.py"))
        if args.limit is not None:
            if args.limit < 0:
                parser.error("--limit must be non-negative")
            model_files = model_files[: args.limit]
            pipeline_files = pipeline_files[: args.limit]
            print(
                f"--limit {args.limit}: checking {len(model_files)} model file(s) "
                f"and {len(pipeline_files)} pipeline file(s).",
                file=sys.stderr,
            )
        for p in model_files:
            targets.append((p, "model"))
        for p in pipeline_files:
            targets.append((p, "pipeline"))

    if args.fix:
        fix_summaries: list[str] = []
        for path, kind in targets:
            for summary in fix_file(path, kind):
                fix_summaries.append(f"{path.relative_to(REPO_ROOT)}: {summary}")
        if fix_summaries:
            print("Removed stale docstring entries:")
            print("\n".join(f"  {s}" for s in fix_summaries))
        else:
            print("No stale docstring entries to remove.")

    all_errors: list[str] = []
    for path, kind in targets:
        all_errors.extend(check_file(path, kind))

    if all_errors:
        print("\n".join(all_errors))
        print(
            f"\nFound {len(all_errors)} docstring/signature mismatch(es).",
            file=sys.stderr,
        )
        if not args.fix and any("documents argument(s) not in the signature" in e for e in all_errors):
            print(
                "Hint: run `python utils/check_forward_call_docstrings.py --fix` "
                "to remove the stale argument entries flagged above. "
                "(Missing-in-docstring entries must be added manually — the tool "
                "never inserts placeholders.)",
                file=sys.stderr,
            )
        return 1

    print("All forward/__call__ arguments are documented.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
