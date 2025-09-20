# fix_imports.py
import os
import sys
import argparse
from typing import List, Tuple

# --- Configuration ---
# The import string to search for, which was incorrectly placed by the previous script.
FAULTY_IMPORT_STRING = "from ..pipeline_utils import"
# ---------------------

def repair_misplaced_imports(directory: str):
    """
    Finds and fixes Python files where an import was incorrectly indented.
    """
    print(f"🔍 Scanning for misplaced imports in '{directory}'...")
    files_repaired = 0

    for root, _, files in os.walk(directory):
        for filename in files:
            if not filename.endswith(".py"):
                continue

            file_path = os.path.join(root, filename)
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                imports_to_move: List[str] = []
                indices_to_remove: List[int] = []

                # First, find any misplaced imports that are indented
                for i, line in enumerate(lines):
                    # Check if the line is indented and contains our faulty import
                    is_indented = line.startswith((' ', '\t'))
                    if is_indented and FAULTY_IMPORT_STRING in line:
                        imports_to_move.append(line.lstrip()) # Store the import without leading whitespace
                        indices_to_remove.append(i)

                # If we found misplaced imports, we need to fix this file
                if not imports_to_move:
                    continue

                print(f"  - ⚠️  Found misplaced import in: {file_path}")

                # Create a new list of lines, excluding the misplaced ones
                cleaned_lines: List[str] = [
                    line for i, line in enumerate(lines) if i not in indices_to_remove
                ]
                
                # Now, find the correct place to insert the imports: after the last top-level import
                insert_index = 0
                for i, line in enumerate(cleaned_lines):
                    # A top-level import is not indented and starts with 'import' or 'from'
                    is_toplevel_import = not line.startswith((' ', '\t')) and line.strip().startswith(('import ', 'from '))
                    if is_toplevel_import:
                        insert_index = i + 1

                # Insert the imports we removed earlier into the correct location
                # We add a newline for clean spacing if needed.
                if insert_index > 0:
                     cleaned_lines.insert(insert_index, '\n')
                     insert_index += 1

                for import_line in reversed(imports_to_move): # Reverse to maintain order when inserting one by one
                    cleaned_lines.insert(insert_index, import_line)

                # Write the repaired content back to the file
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.writelines(cleaned_lines)
                
                print(f"  - ✅ Repaired {file_path}")
                files_repaired += 1

            except Exception as e:
                print(f"  - ❌ Failed to process {file_path}: {e}", file=sys.stderr)

    if files_repaired == 0:
        print("\n✅ No files with misplaced imports were found.")
    else:
        print(f"\n✨ Repair complete. Total files fixed: {files_repaired}.")


def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(
        description="Corrects misplaced, indented imports created by a previous script.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "directory",
        type=str,
        help="The target directory to scan, e.g., 'src/diffusers/pipelines'."
    )
    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        print(f"❌ Error: Directory not found at '{args.directory}'", file=sys.stderr)
        sys.exit(1)

    repair_misplaced_imports(args.directory)


if __name__ == "__main__":
    main()