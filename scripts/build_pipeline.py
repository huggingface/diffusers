"""Run a pipeline assembly recipe; component tensor mappings live in diffusers.loaders.conversion."""

import argparse
import runpy
import sys
from pathlib import Path


def main():
    directory = Path(__file__).resolve().parent / "recipes"
    recipes = {path.stem: path for path in directory.glob("*.py") if not path.name.startswith("_")}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recipe", nargs="?", choices=sorted(recipes))
    parser.add_argument("--list", action="store_true", help="List available recipes")
    parser.add_argument("recipe_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.list:
        print("\n".join(sorted(recipes)))
        return
    if not args.recipe:
        parser.error("Choose a recipe, or use --list")
    previous_argv, previous_path = sys.argv[:], sys.path[:]
    try:
        sys.argv = [str(recipes[args.recipe]), *args.recipe_args]
        sys.path.insert(0, str(directory))
        runpy.run_path(str(recipes[args.recipe]), run_name="__main__")
    finally:
        sys.argv[:] = previous_argv
        sys.path[:] = previous_path


if __name__ == "__main__":
    main()
