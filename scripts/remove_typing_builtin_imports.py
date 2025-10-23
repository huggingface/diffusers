#!/usr/bin/env python3
"""
Remove lower-case built-in generics imported from `typing`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Iterator, Sequence


try:
    import libcst as cst
except ImportError as exc:  # pragma: no cover - dependency guard
    raise SystemExit("This script requires `libcst`. Install it via `pip install libcst` and retry.") from exc


BUILTIN_TYPING_NAMES = frozenset({"callable", "dict", "frozenset", "list", "set", "tuple", "type"})


class TypingBuiltinImportRemover(cst.CSTTransformer):
    def __init__(self) -> None:
        self.changed = False
        self.removed: list[str] = []
        self.warnings: list[str] = []

    def leave_ImportFrom(self, original_node: cst.ImportFrom, updated_node: cst.ImportFrom) -> cst.BaseStatement:
        module_name = self._module_name(updated_node.module)
        if module_name != "typing":
            return updated_node

        names = updated_node.names
        if isinstance(names, cst.ImportStar):
            self.warnings.append("encountered `from typing import *` (skipped)")
            return updated_node

        new_aliases = []
        removed_here: list[str] = []
        for alias in names:
            if isinstance(alias, cst.ImportStar):
                self.warnings.append("encountered `from typing import *` (skipped)")
                return updated_node
            if not isinstance(alias.name, cst.Name):
                new_aliases.append(alias)
                continue
            imported_name = alias.name.value
            if imported_name in BUILTIN_TYPING_NAMES:
                removed_here.append(imported_name)
                continue
            new_aliases.append(alias)

        if not removed_here:
            return updated_node

        self.changed = True
        self.removed.extend(removed_here)

        if not new_aliases:
            return cst.RemoveFromParent()
        # Ensure trailing commas are removed.
        formatted_aliases = []
        for alias in new_aliases:
            if alias.comma is not None and alias is new_aliases[-1]:
                formatted_aliases.append(alias.with_changes(comma=None))
            else:
                formatted_aliases.append(alias)

        return updated_node.with_changes(names=tuple(formatted_aliases))

    def _module_name(self, node: cst.BaseExpression | None) -> str | None:
        if node is None:
            return None
        if isinstance(node, cst.Name):
            return node.value
        if isinstance(node, cst.Attribute):
            prefix = self._module_name(node.value)
            if prefix is None:
                return node.attr.value
            return f"{prefix}.{node.attr.value}"
        return None


def iter_python_files(paths: Iterable[Path]) -> Iterator[Path]:
    for path in paths:
        if path.is_dir():
            yield from (p for p in path.rglob("*.py") if not p.name.startswith("."))
            yield from (p for p in path.rglob("*.pyi") if not p.name.startswith("."))
        elif path.suffix in {".py", ".pyi"}:
            yield path


def process_file(path: Path, dry_run: bool) -> tuple[bool, TypingBuiltinImportRemover]:
    source = path.read_text(encoding="utf-8")
    module = cst.parse_module(source)
    transformer = TypingBuiltinImportRemover()
    updated = module.visit(transformer)

    if not transformer.changed or source == updated.code:
        return False, transformer

    if not dry_run:
        path.write_text(updated.code, encoding="utf-8")
    return True, transformer


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Remove lower-case built-in generics imported from typing.")
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[Path("src")],
        help="Files or directories to rewrite (default: src).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report files that would change without writing them.",
    )
    args = parser.parse_args(argv)

    files = sorted(set(iter_python_files(args.paths)))
    if not files:
        print("No Python files matched the provided paths.", file=sys.stderr)
        return 1

    changed_any = False
    for path in files:
        changed, transformer = process_file(path, dry_run=args.dry_run)
        if changed:
            changed_any = True
            action = "Would update" if args.dry_run else "Updated"
            removed = ", ".join(sorted(set(transformer.removed)))
            print(f"{action}: {path} (removed typing imports: {removed})")
        for warning in transformer.warnings:
            print(f"Warning: {path}: {warning}", file=sys.stderr)

    if not changed_any:
        print("No changes needed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
