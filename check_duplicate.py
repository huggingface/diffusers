import ast
import argparse
from pathlib import Path
from collections import defaultdict

# This script requires Python 3.9+ for the ast.unparse() function.

class ClassMethodVisitor(ast.NodeVisitor):
    """
    An AST visitor that collects method names and the source code of their bodies.
    """
    def __init__(self):
        self.methods = defaultdict(list)

    def visit_ClassDef(self, node: ast.ClassDef):
        """
        Visits a class definition, then inspects its methods.
        """
        class_name = node.name
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                method_name = item.name
                body_source = ast.unparse(item.body).strip()
                self.methods[method_name].append((class_name, body_source))
        self.generic_visit(node)

def find_duplicate_method_content(directory: str, show_code: bool = True):
    """
    Parses all Python files in a directory to find methods with duplicate content.

    Args:
        directory: The path to the directory to inspect.
        show_code: If True, prints the shared code block for each duplicate.
    """
    target_dir = Path(directory)
    if not target_dir.is_dir():
        print(f"❌ Error: '{directory}' is not a valid directory.")
        return

    visitor = ClassMethodVisitor()

    for py_file in target_dir.rglob('*.py'):
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                source_code = f.read()
                tree = ast.parse(source_code, filename=py_file)
                visitor.visit(tree)
        except Exception as e:
            print(f"⚠️ Warning: Could not process {py_file}. Error: {e}")

    print("\n--- Duplicate Method Content Report ---")
    duplicates_found = False

    for method_name, implementations in sorted(visitor.methods.items()):
        body_groups = defaultdict(list)
        for class_name, body_source in implementations:
            body_groups[body_source].append(class_name)

        for body_source, class_list in body_groups.items():
            if len(class_list) > 1:
                duplicates_found = True
                unique_classes = sorted(list(set(class_list)))
                print(f"\n[+] Method `def {method_name}(...)` has identical content in {len(unique_classes)} classes:")
                for class_name in unique_classes:
                    print(f"  - {class_name}")

                # Conditionally print the shared code block based on the flag
                if show_code:
                    print("\n  Shared Code Block:")
                    indented_code = "\n".join([f"    {line}" for line in body_source.splitlines()])
                    print(indented_code)
                    print("  " + "-" * 30)

    if not duplicates_found:
        print("\n✅ No methods with identical content were found across classes.")

def main():
    """Main function to set up argument parsing."""
    parser = argparse.ArgumentParser(
        description="Find methods with identical content across Python classes in a directory."
    )
    parser.add_argument(
        "directory",
        type=str,
        help="The path to the directory to inspect."
    )
    # New argument to control output verbosity
    parser.add_argument(
        "--hide-code",
        action="store_true",
        help="Do not print the shared code block for each duplicate found."
    )
    args = parser.parse_args()
    find_duplicate_method_content(args.directory, show_code=not args.hide_code)

if __name__ == "__main__":
    main()