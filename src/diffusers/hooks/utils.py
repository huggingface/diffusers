import ast
import inspect
import textwrap
from typing import List


def _extract_return_information(func) -> List[str]:
    """Extracts return variable names in order from a function."""
    try:
        source = inspect.getsource(func)
        source = textwrap.dedent(source)  # Modify indentation to make parsing compatible
    except (OSError, TypeError):
        try:
            source_file = inspect.getfile(func)
            with open(source_file, "r", encoding="utf-8") as f:
                source = f.read()

            # Extract function definition manually
            source_lines = source.splitlines()
            func_name = func.__name__
            start_line = None
            indent_level = None
            extracted_lines = []

            for i, line in enumerate(source_lines):
                stripped = line.strip()
                if stripped.startswith(f"def {func_name}("):
                    start_line = i
                    indent_level = len(line) - len(line.lstrip())
                    extracted_lines.append(line)
                    continue

                if start_line is not None:
                    # Stop when indentation level decreases (end of function)
                    current_indent = len(line) - len(line.lstrip())
                    if current_indent <= indent_level and line.strip():
                        break
                    extracted_lines.append(line)

            source = "\n".join(extracted_lines)
        except Exception as e:
            raise RuntimeError(f"Failed to retrieve function source: {e}")

    # Parse source code using AST
    tree = ast.parse(source)
    return_vars = []

    class ReturnVisitor(ast.NodeVisitor):
        def visit_Return(self, node):
            if isinstance(node.value, ast.Tuple):
                # Multiple return values
                return_vars.extend(var.id for var in node.value.elts if isinstance(var, ast.Name))
            elif isinstance(node.value, ast.Name):
                # Single return value
                return_vars.append(node.value.id)

    visitor = ReturnVisitor()
    visitor.visit(tree)
    return return_vars
