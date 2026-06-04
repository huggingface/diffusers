import ast
import json
import sys


SRC_DIRS = ["src/diffusers/pipelines/", "src/diffusers/models/", "src/diffusers/schedulers/"]
MIXIN_BASES = {"ModelMixin", "SchedulerMixin", "DiffusionPipeline"}


def extract_classes_from_file(filepath: str) -> list[str]:
    with open(filepath) as f:
        tree = ast.parse(f.read())

    classes = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        base_names = set()
        for base in node.bases:
            if isinstance(base, ast.Name):
                base_names.add(base.id)
            elif isinstance(base, ast.Attribute):
                base_names.add(base.attr)
        if base_names & MIXIN_BASES:
            classes.append(node.name)

    return classes


def extract_imports_from_file(filepath: str) -> set[str]:
    with open(filepath) as f:
        tree = ast.parse(f.read())

    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[-1])

    return names


def main():
    pr_files = json.load(sys.stdin)

    new_classes = []
    for f in pr_files:
        if f["status"] != "added" or not f["filename"].endswith(".py"):
            continue
        if not any(f["filename"].startswith(d) for d in SRC_DIRS):
            continue
        try:
            new_classes.extend(extract_classes_from_file(f["filename"]))
        except (FileNotFoundError, SyntaxError):
            continue

    if not new_classes:
        sys.exit(0)

    new_test_files = [
        f["filename"]
        for f in pr_files
        if f["status"] == "added" and f["filename"].startswith("tests/") and f["filename"].endswith(".py")
    ]

    imported_names = set()
    for filepath in new_test_files:
        try:
            imported_names |= extract_imports_from_file(filepath)
        except (FileNotFoundError, SyntaxError):
            continue

    untested = [cls for cls in new_classes if cls not in imported_names]

    if untested:
        print(f"missing-tests: {', '.join(untested)}")
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
