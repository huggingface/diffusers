#!/usr/bin/env python3
"""
Update copyright year in source files to the current year.
"""

import re
import sys
from datetime import datetime
from pathlib import Path


# Set current year
YEAR = str(datetime.now().year)

EXTENSIONS = {".py", ".md"}

# Directories to exclude
EXCLUDE_DIRS = {
    ".venv",
    "venv",
    "node_modules",
    ".git",
    "__pycache__",
    ".tox",
    "dist",
    "build",
    ".egg-info",
}

# Regex patterns to match copyright lines
COPYRIGHT_PATTERNS = [
    re.compile(r"(# Copyright )(\d{4})( The HuggingFace Team\.)"),
    re.compile(r"(# Copyright \(c\) )(\d{4})( The HuggingFace Team\.)"),
    re.compile(r"(Copyright )(\d{4})( The HuggingFace Team\.)"),
    re.compile(r"(Copyright \(c\) )(\d{4})( The HuggingFace Team\.)"),
]


def should_exclude(path: Path) -> bool:
    """Check if path should be excluded."""
    return any(excluded in path.parts for excluded in EXCLUDE_DIRS)


def update_file(file_path: Path) -> int:
    """Update copyright year in a single file. Returns number of updates."""
    try:
        content = file_path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"Warning: Could not read {file_path}: {e}", file=sys.stderr)
        return 0

    new_content = content
    total_count = 0

    for pattern in COPYRIGHT_PATTERNS:
        new_content, count = pattern.subn(rf"\g<1>{YEAR}\g<3>", new_content)
        total_count += count

    if total_count > 0:
        try:
            file_path.write_text(new_content, encoding="utf-8")
            print(f"✓ Updated {total_count} line(s) in {file_path}")
        except Exception as e:
            print(f"Error: Could not write {file_path}: {e}", file=sys.stderr)
            return 0

    return total_count


def main():
    repo_root = Path(".").resolve()

    print(f"Updating copyright to {YEAR}...")

    total_files = 0
    total_updates = 0

    for file_path in repo_root.rglob("*"):
        if not file_path.is_file() or should_exclude(file_path):
            continue

        if file_path.suffix in EXTENSIONS:
            updates = update_file(file_path)
            if updates > 0:
                total_files += 1
                total_updates += updates

    print(f"\nSummary: Updated {total_updates} line(s) in {total_files} file(s)")

    # Exit with 0 if updates were made, 1 if no updates needed
    return 0 if total_updates > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
