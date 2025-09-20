# add_imports_v2.py
import os
import sys
import argparse
from typing import List, Dict, Set

# --- Configuration ---
METHODS_TO_FIND: Set[str] = {'retrieve_timesteps', 'retrieve_latents', 'calculate_shift'}
IMPORT_SOURCE: str = "..pipeline_utils"
# This is the specific file to exclude from any modifications.
FILE_TO_EXCLUDE: str = "src/diffusers/pipelines/pipeline_utils.py"
# ---------------------


def find_relevant_files(directory: str) -> Dict[str, List[str]]:
    """
    Scans a directory for Python files containing any of the target methods.

    Args:
        directory: The path to the directory to scan.

    Returns:
        A dictionary mapping file paths to a list of found methods in that file.
    """
    print(f"🔍 Scanning for files in '{directory}'...")
    files_to_modify: Dict[str, List[str]] = {}
    excluded_file_path = os.path.normpath(FILE_TO_EXCLUDE)

    for root, _, files in os.walk(directory):
        for filename in files:
            if not filename.endswith(".py"):
                continue

            file_path = os.path.join(root, filename)

            # --- MODIFICATION ---
            # Check if the current file is the one we need to exclude.
            if os.path.normpath(file_path) == excluded_file_path:
                continue
            # --- END MODIFICATION ---

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                found_methods = [method for method in METHODS_TO_FIND if method in content]

                if found_methods:
                    # Sort for consistent import order
                    files_to_modify[file_path] = sorted(found_methods)
            except Exception as e:
                print(f"Could not read {file_path}: {e}", file=sys.stderr)

    return files_to_modify


def modify_files(files_to_process: Dict[str, List[str]]):
    """
    Adds the required import statement to the top of each file.

    Args:
        files_to_process: A dictionary of file paths and the methods they contain.
    """
    if not files_to_process:
        print("✅ No files required modification.")
        return

    print("\n✍️  Adding imports to the following files:")
    for file_path, methods in files_to_process.items():
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # Construct the new import statement
            methods_str = ", ".join(methods)
            import_statement = f"from {IMPORT_SOURCE} import {methods_str}\n"

            # Avoid adding the exact same import statement if it already exists
            if any(line.strip() == import_statement.strip() for line in lines):
                print(f"  - ⚠️  Skipping {file_path} (import already exists)")
                continue

            # Find the best place to insert the new import.
            # We'll place it after the last existing top-level import.
            insert_index = 0
            for i, line in enumerate(lines):
                stripped_line = line.strip()
                if stripped_line.startswith(('import ', 'from ')):
                    insert_index = i + 1
            
            # Insert a newline for spacing if we're adding after other imports
            if insert_index > 0 and lines[insert_index-1].strip() != "":
                 lines.insert(insert_index, '\n')
                 insert_index += 1

            lines.insert(insert_index, import_statement)

            # Write the changes back to the file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)

            print(f"  - ✅ Modified {file_path}")

        except Exception as e:
            print(f"  - ❌ Failed to modify {file_path}: {e}", file=sys.stderr)


def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(
        description="Finds files with specific method mentions and adds corresponding imports.",
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

    # 1. Find files and the methods they contain
    found_files = find_relevant_files(args.directory)

    # 2. Enlist the files that will be modified
    if found_files:
        print("\n--- Files Found ---")
        for path, methods in found_files.items():
            print(f"- {path}\n    (Contains: {', '.join(methods)})")
        print("-------------------\n")
    else:
        print("✅ No files found containing the specified methods.")
        return

    # 3. Modify the enlisted files
    modify_files(found_files)
    print("\n✨ Script finished.")


if __name__ == "__main__":
    main()