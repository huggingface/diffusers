import ast
import importlib
import inspect
import textwrap


class ReturnNameVisitor(ast.NodeVisitor):
    """Thanks to ChatGPT for pairing."""

    def __init__(self):
        self.return_names = []

    def visit_Return(self, node):
        # Check if the return value is a tuple.
        if isinstance(node.value, ast.Tuple):
            for elt in node.value.elts:
                if isinstance(elt, ast.Name):
                    self.return_names.append(elt.id)
                else:
                    try:
                        self.return_names.append(ast.unparse(elt))
                    except Exception:
                        self.return_names.append(str(elt))
        else:
            if isinstance(node.value, ast.Name):
                self.return_names.append(node.value.id)
            else:
                try:
                    self.return_names.append(ast.unparse(node.value))
                except Exception:
                    self.return_names.append(str(node.value))
        self.generic_visit(node)

    def _determine_parent_module(self, cls):
        from diffusers import DiffusionPipeline
        from diffusers.models.modeling_utils import ModelMixin

        if issubclass(cls, DiffusionPipeline):
            return "pipelines"
        elif issubclass(cls, ModelMixin):
            return "models"
        else:
            raise NotImplementedError

    def get_ast_tree(self, cls, attribute_name="encode_prompt"):
        parent_module_name = self._determine_parent_module(cls)
        main_module = importlib.import_module(f"diffusers.{parent_module_name}")
        current_cls_module = getattr(main_module, cls.__name__)
        source_code = inspect.getsource(getattr(current_cls_module, attribute_name))
        source_code = textwrap.dedent(source_code)
        tree = ast.parse(source_code)
        return tree
