#!/usr/bin/env python3
"""
Fix F631-style asserts of the form:
    assert (<expr>, "message")
…into:
    assert <expr>, "message"

Scans recursively under tests/lora/.

Usage:
    python fix_assert_tuple.py [--root tests/lora] [--dry-run]
"""

import argparse
import ast
from pathlib import Path
from typing import Tuple, List, Optional


class AssertTupleFixer(ast.NodeTransformer):
    """
    Transform `assert (<expr>, <msg>)` into `assert <expr>, <msg>`.
    We only rewrite when the assert test is a Tuple with exactly 2 elements.
    """
    def __init__(self):
        super().__init__()
        self.fixed_locs: List[Tuple[int, int]] = []

    def visit_Assert(self, node: ast.Assert) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.test, ast.Tuple) and len(node.test.elts) == 2:
            cond, msg = node.test.elts
            # Convert only if this *looks* like a real assert-with-message tuple,
            # i.e. keep anything as msg (string, f-string, name, call, etc.)
            new_node = ast.Assert(test=cond, msg=msg)
            ast.copy_location(new_node, node)
            ast.fix_missing_locations(new_node)
            self.fixed_locs.append((node.lineno, node.col_offset))
            return new_node
        return node


def fix_file(path: Path, dry_run: bool = False) -> int:
    """
    Returns number of fixes applied.
    """
    try:
        src = path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"Could not read {path}: {e}")
        return 0

    try:
        tree = ast.parse(src, filename=str(path))
    except SyntaxError:
        # Skip files that don’t parse (partial edits, etc.)
        return 0

    fixer = AssertTupleFixer()
    new_tree = fixer.visit(tree)
    fixes = len(fixer.fixed_locs)
    if fixes == 0:
        return 0

    try:
        new_src = ast.unparse(new_tree)  # Python 3.9+
    except Exception as e:
        print(f"Failed to unparse {path}: {e}")
        return 0

    if dry_run:
        for (lineno, col) in fixer.fixed_locs:
            print(f"[DRY-RUN] {path}:{lineno}:{col} -> fixed assert tuple")
        return fixes

    # Backup and write
    backup = path.with_suffix(path.suffix + ".bak")
    try:
        if not backup.exists():
            backup.write_text(src, encoding="utf-8")
        path.write_text(new_src, encoding="utf-8")
        for (lineno, col) in fixer.fixed_locs:
            print(f"Fixed {path}:{lineno}:{col}")
    except Exception as e:
        print(f"Failed to write {path}: {e}")
        return 0

    return fixes


def main():
    ap = argparse.ArgumentParser(description="Fix F631-style tuple asserts.")
    ap.add_argument("--root", default="tests/lora", help="Root directory to scan")
    ap.add_argument("--dry-run", action="store_true", help="Report changes but don't write")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"{root} does not exist.")
        return

    total_files = 0
    total_fixes = 0
    for pyfile in root.rglob("*.py"):
        total_files += 1
        total_fixes += fix_file(pyfile, dry_run=args.dry_run)

    print(f"\nScanned {total_files} file(s). Applied {total_fixes} fix(es).")
    if args.dry_run:
        print("Run again without --dry-run to apply changes.")


if __name__ == "__main__":
    main()
