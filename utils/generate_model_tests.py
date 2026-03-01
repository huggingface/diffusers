#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 HuggingFace Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Utility script to generate test suites for diffusers model classes.

Usage:
    python utils/generate_model_tests.py src/diffusers/models/transformers/transformer_flux.py

This will analyze the model file and generate a test file with appropriate
test classes based on the model's mixins and attributes.
"""

import argparse
import ast
import sys
from pathlib import Path


MIXIN_TO_TESTER = {
    "ModelMixin": "ModelTesterMixin",
    "PeftAdapterMixin": "LoraTesterMixin",
}

ATTRIBUTE_TO_TESTER = {
    "_cp_plan": "ContextParallelTesterMixin",
    "_supports_gradient_checkpointing": "TrainingTesterMixin",
}

ALWAYS_INCLUDE_TESTERS = [
    "ModelTesterMixin",
    "MemoryTesterMixin",
    "TorchCompileTesterMixin",
]

# Attention-related class names that indicate the model uses attention
ATTENTION_INDICATORS = {
    "AttentionMixin",
    "AttentionModuleMixin",
}

OPTIONAL_TESTERS = [
    # Quantization testers
    ("BitsAndBytesTesterMixin", "bnb"),
    ("QuantoTesterMixin", "quanto"),
    ("TorchAoTesterMixin", "torchao"),
    ("GGUFTesterMixin", "gguf"),
    ("ModelOptTesterMixin", "modelopt"),
    # Quantization compile testers
    ("BitsAndBytesCompileTesterMixin", "bnb_compile"),
    ("QuantoCompileTesterMixin", "quanto_compile"),
    ("TorchAoCompileTesterMixin", "torchao_compile"),
    ("GGUFCompileTesterMixin", "gguf_compile"),
    ("ModelOptCompileTesterMixin", "modelopt_compile"),
    # Cache testers
    ("PyramidAttentionBroadcastTesterMixin", "pab_cache"),
    ("FirstBlockCacheTesterMixin", "fbc_cache"),
    ("FasterCacheTesterMixin", "faster_cache"),
    # Other testers
    ("SingleFileTesterMixin", "single_file"),
    ("IPAdapterTesterMixin", "ip_adapter"),
]


class ModelAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.model_classes = []
        self.current_class = None
        self.imports = set()

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            self.imports.add(alias.name.split(".")[-1])
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        for alias in node.names:
            self.imports.add(alias.name)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef):
        base_names = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                base_names.append(base.id)
            elif isinstance(base, ast.Attribute):
                base_names.append(base.attr)

        if "ModelMixin" in base_names:
            class_info = {
                "name": node.name,
                "bases": base_names,
                "attributes": {},
                "has_forward": False,
                "init_params": [],
            }

            for item in node.body:
                if isinstance(item, ast.Assign):
                    for target in item.targets:
                        if isinstance(target, ast.Name):
                            attr_name = target.id
                            if attr_name.startswith("_"):
                                class_info["attributes"][attr_name] = self._get_value(item.value)

                elif isinstance(item, ast.FunctionDef):
                    if item.name == "forward":
                        class_info["has_forward"] = True
                        class_info["forward_params"] = self._extract_func_params(item)
                    elif item.name == "__init__":
                        class_info["init_params"] = self._extract_func_params(item)

            self.model_classes.append(class_info)

        self.generic_visit(node)

    def _extract_func_params(self, func_node: ast.FunctionDef) -> list[dict]:
        params = []
        args = func_node.args

        num_defaults = len(args.defaults)
        num_args = len(args.args)
        first_default_idx = num_args - num_defaults

        for i, arg in enumerate(args.args):
            if arg.arg == "self":
                continue

            param_info = {"name": arg.arg, "type": None, "default": None}

            if arg.annotation:
                param_info["type"] = self._get_annotation_str(arg.annotation)

            default_idx = i - first_default_idx
            if default_idx >= 0 and default_idx < len(args.defaults):
                param_info["default"] = self._get_value(args.defaults[default_idx])

            params.append(param_info)

        return params

    def _get_annotation_str(self, node) -> str:
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Constant):
            return repr(node.value)
        elif isinstance(node, ast.Subscript):
            base = self._get_annotation_str(node.value)
            if isinstance(node.slice, ast.Tuple):
                args = ", ".join(self._get_annotation_str(el) for el in node.slice.elts)
            else:
                args = self._get_annotation_str(node.slice)
            return f"{base}[{args}]"
        elif isinstance(node, ast.Attribute):
            return f"{self._get_annotation_str(node.value)}.{node.attr}"
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            left = self._get_annotation_str(node.left)
            right = self._get_annotation_str(node.right)
            return f"{left} | {right}"
        elif isinstance(node, ast.Tuple):
            return ", ".join(self._get_annotation_str(el) for el in node.elts)
        return "Any"

    def _get_value(self, node):
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Name):
            if node.id == "None":
                return None
            elif node.id == "True":
                return True
            elif node.id == "False":
                return False
            return node.id
        elif isinstance(node, ast.List):
            return [self._get_value(el) for el in node.elts]
        elif isinstance(node, ast.Dict):
            return {self._get_value(k): self._get_value(v) for k, v in zip(node.keys, node.values)}
        return "<complex>"


def analyze_model_file(filepath: str) -> tuple[list[dict], set[str]]:
    with open(filepath) as f:
        source = f.read()

    tree = ast.parse(source)
    analyzer = ModelAnalyzer()
    analyzer.visit(tree)

    return analyzer.model_classes, analyzer.imports


def determine_testers(model_info: dict, include_optional: list[str], imports: set[str]) -> list[str]:
    testers = list(ALWAYS_INCLUDE_TESTERS)

    for base in model_info["bases"]:
        if base in MIXIN_TO_TESTER:
            tester = MIXIN_TO_TESTER[base]
            if tester not in testers:
                testers.append(tester)

    for attr, tester in ATTRIBUTE_TO_TESTER.items():
        if attr in model_info["attributes"]:
            value = model_info["attributes"][attr]
            if value is not None and value is not False:
                if tester not in testers:
                    testers.append(tester)

    if "_cp_plan" in model_info["attributes"] and model_info["attributes"]["_cp_plan"] is not None:
        if "ContextParallelTesterMixin" not in testers:
            testers.append("ContextParallelTesterMixin")

    # Include AttentionTesterMixin if the model imports attention-related classes
    if imports & ATTENTION_INDICATORS:
        testers.append("AttentionTesterMixin")

    for tester, flag in OPTIONAL_TESTERS:
        if flag in include_optional:
            if tester not in testers:
                testers.append(tester)

    return testers


def generate_config_class(model_info: dict, model_name: str) -> str:
    class_name = f"{model_name}TesterConfig"
    model_class = model_info["name"]
    forward_params = model_info.get("forward_params", [])
    init_params = model_info.get("init_params", [])

    lines = [
        f"class {class_name}:",
        "    @property",
        "    def model_class(self):",
        f"        return {model_class}",
        "",
        "    @property",
        "    def pretrained_model_name_or_path(self):",
        '        return ""  # TODO: Set Hub repository ID',
        "",
        "    @property",
        "    def pretrained_model_kwargs(self):",
        '        return {"subfolder": "transformer"}',
        "",
        "    @property",
        "    def generator(self):",
        '        return torch.Generator("cpu").manual_seed(0)',
        "",
        "    def get_init_dict(self) -> dict[str, int | list[int]]:",
    ]

    if init_params:
        lines.append("        # __init__ parameters:")
        for param in init_params:
            type_str = f": {param['type']}" if param["type"] else ""
            default_str = f" = {param['default']}" if param["default"] is not None else ""
            lines.append(f"        #   {param['name']}{type_str}{default_str}")

    lines.extend(
        [
            "        return {}",
            "",
            "    def get_dummy_inputs(self) -> dict[str, torch.Tensor]:",
        ]
    )

    if forward_params:
        lines.append("        # forward() parameters:")
        for param in forward_params:
            type_str = f": {param['type']}" if param["type"] else ""
            default_str = f" = {param['default']}" if param["default"] is not None else ""
            lines.append(f"        #   {param['name']}{type_str}{default_str}")

    lines.extend(
        [
            "        # TODO: Fill in dummy inputs",
            "        return {}",
            "",
            "    @property",
            "    def input_shape(self) -> tuple[int, ...]:",
            "        return (1, 1)",
            "",
            "    @property",
            "    def output_shape(self) -> tuple[int, ...]:",
            "        return (1, 1)",
        ]
    )

    return "\n".join(lines)


def generate_test_class(model_name: str, config_class: str, tester: str) -> str:
    tester_short = tester.replace("TesterMixin", "")
    class_name = f"Test{model_name}{tester_short}"

    lines = [f"class {class_name}({config_class}, {tester}):"]

    if tester == "TorchCompileTesterMixin":
        lines.extend(
            [
                "    @property",
                "    def different_shapes_for_compilation(self):",
                "        return [(4, 4), (4, 8), (8, 8)]",
                "",
                "    def get_dummy_inputs(self, height: int = 4, width: int = 4) -> dict[str, torch.Tensor]:",
                "        # TODO: Implement dynamic input generation",
                "        return {}",
            ]
        )
    elif tester == "IPAdapterTesterMixin":
        lines.extend(
            [
                "    @property",
                "    def ip_adapter_processor_cls(self):",
                "        return None  # TODO: Set processor class",
                "",
                "    def modify_inputs_for_ip_adapter(self, model, inputs_dict):",
                "        # TODO: Add IP adapter image embeds to inputs",
                "        return inputs_dict",
                "",
                "    def create_ip_adapter_state_dict(self, model):",
                "        # TODO: Create IP adapter state dict",
                "        return {}",
            ]
        )
    elif tester == "SingleFileTesterMixin":
        lines.extend(
            [
                "    @property",
                "    def ckpt_path(self):",
                '        return ""  # TODO: Set checkpoint path',
                "",
                "    @property",
                "    def alternate_ckpt_paths(self):",
                "        return []",
                "",
                "    @property",
                "    def pretrained_model_name_or_path(self):",
                '        return ""  # TODO: Set Hub repository ID',
            ]
        )
    elif tester == "GGUFTesterMixin":
        lines.extend(
            [
                "    @property",
                "    def gguf_filename(self):",
                '        return ""  # TODO: Set GGUF filename',
                "",
                "    def get_dummy_inputs(self) -> dict[str, torch.Tensor]:",
                "        # TODO: Override with larger inputs for quantization tests",
                "        return {}",
            ]
        )
    elif tester in ["BitsAndBytesTesterMixin", "QuantoTesterMixin", "TorchAoTesterMixin", "ModelOptTesterMixin"]:
        lines.extend(
            [
                "    def get_dummy_inputs(self) -> dict[str, torch.Tensor]:",
                "        # TODO: Override with larger inputs for quantization tests",
                "        return {}",
            ]
        )
    elif tester in [
        "BitsAndBytesCompileTesterMixin",
        "QuantoCompileTesterMixin",
        "TorchAoCompileTesterMixin",
        "ModelOptCompileTesterMixin",
    ]:
        lines.extend(
            [
                "    def get_dummy_inputs(self) -> dict[str, torch.Tensor]:",
                "        # TODO: Override with larger inputs for quantization compile tests",
                "        return {}",
            ]
        )
    elif tester == "GGUFCompileTesterMixin":
        lines.extend(
            [
                "    @property",
                "    def gguf_filename(self):",
                '        return ""  # TODO: Set GGUF filename',
                "",
                "    def get_dummy_inputs(self) -> dict[str, torch.Tensor]:",
                "        # TODO: Override with larger inputs for quantization compile tests",
                "        return {}",
            ]
        )
    elif tester in [
        "PyramidAttentionBroadcastTesterMixin",
        "FirstBlockCacheTesterMixin",
        "FasterCacheTesterMixin",
    ]:
        lines.append("    pass")
    elif tester == "LoraHotSwappingForModelTesterMixin":
        lines.extend(
            [
                "    @property",
                "    def different_shapes_for_compilation(self):",
                "        return [(4, 4), (4, 8), (8, 8)]",
                "",
                "    def get_dummy_inputs(self, height: int = 4, width: int = 4) -> dict[str, torch.Tensor]:",
                "        # TODO: Implement dynamic input generation",
                "        return {}",
            ]
        )
    else:
        lines.append("    pass")

    return "\n".join(lines)


def generate_test_file(model_info: dict, model_filepath: str, include_optional: list[str], imports: set[str]) -> str:
    model_name = model_info["name"].replace("2DModel", "").replace("3DModel", "").replace("Model", "")
    testers = determine_testers(model_info, include_optional, imports)
    tester_imports = sorted(set(testers) - {"LoraHotSwappingForModelTesterMixin"})

    lines = [
        "# coding=utf-8",
        "# Copyright 2025 HuggingFace Inc.",
        "#",
        '# Licensed under the Apache License, Version 2.0 (the "License");',
        "# you may not use this file except in compliance with the License.",
        "# You may obtain a copy of the License at",
        "#",
        "#     http://www.apache.org/licenses/LICENSE-2.0",
        "#",
        "# Unless required by applicable law or agreed to in writing, software",
        '# distributed under the License is distributed on an "AS IS" BASIS,',
        "# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.",
        "# See the License for the specific language governing permissions and",
        "# limitations under the License.",
        "",
        "import torch",
        "",
        f"from diffusers import {model_info['name']}",
        "from diffusers.utils.torch_utils import randn_tensor",
        "",
        "from ...testing_utils import enable_full_determinism, torch_device",
    ]

    if "LoraTesterMixin" in testers:
        lines.append("from ..test_modeling_common import LoraHotSwappingForModelTesterMixin")

    lines.extend(
        [
            "from ..testing_utils import (",
            *[f"    {tester}," for tester in sorted(tester_imports)],
            ")",
            "",
            "",
            "enable_full_determinism()",
            "",
            "",
        ]
    )

    config_class = f"{model_name}TesterConfig"
    lines.append(generate_config_class(model_info, model_name))
    lines.append("")
    lines.append("")

    for tester in testers:
        lines.append(generate_test_class(model_name, config_class, tester))
        lines.append("")
        lines.append("")

    if "LoraTesterMixin" in testers:
        lines.append(generate_test_class(model_name, config_class, "LoraHotSwappingForModelTesterMixin"))
        lines.append("")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def get_test_output_path(model_filepath: str) -> str:
    path = Path(model_filepath)
    model_filename = path.stem

    if "transformers" in path.parts:
        return f"tests/models/transformers/test_models_{model_filename}.py"
    elif "unets" in path.parts:
        return f"tests/models/unets/test_models_{model_filename}.py"
    elif "autoencoders" in path.parts:
        return f"tests/models/autoencoders/test_models_{model_filename}.py"
    else:
        return f"tests/models/test_models_{model_filename}.py"


def main():
    parser = argparse.ArgumentParser(description="Generate test suite for a diffusers model class")
    parser.add_argument(
        "model_filepath",
        type=str,
        help="Path to the model file (e.g., src/diffusers/models/transformers/transformer_flux.py)",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None, help="Output file path (default: auto-generated based on model path)"
    )
    parser.add_argument(
        "--include",
        "-i",
        type=str,
        nargs="*",
        default=[],
        choices=[
            "bnb",
            "quanto",
            "torchao",
            "gguf",
            "modelopt",
            "bnb_compile",
            "quanto_compile",
            "torchao_compile",
            "gguf_compile",
            "modelopt_compile",
            "pab_cache",
            "fbc_cache",
            "faster_cache",
            "single_file",
            "ip_adapter",
            "all",
        ],
        help="Optional testers to include",
    )
    parser.add_argument(
        "--class-name",
        "-c",
        type=str,
        default=None,
        help="Specific model class to generate tests for (default: first model class found)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print generated code without writing to file")

    args = parser.parse_args()

    if not Path(args.model_filepath).exists():
        print(f"Error: File not found: {args.model_filepath}", file=sys.stderr)
        sys.exit(1)

    model_classes, imports = analyze_model_file(args.model_filepath)

    if not model_classes:
        print(f"Error: No model classes found in {args.model_filepath}", file=sys.stderr)
        sys.exit(1)

    if args.class_name:
        model_info = next((m for m in model_classes if m["name"] == args.class_name), None)
        if not model_info:
            available = [m["name"] for m in model_classes]
            print(f"Error: Class '{args.class_name}' not found. Available: {available}", file=sys.stderr)
            sys.exit(1)
    else:
        model_info = model_classes[0]
        if len(model_classes) > 1:
            print(f"Multiple model classes found, using: {model_info['name']}", file=sys.stderr)
            print("Use --class-name to specify a different class", file=sys.stderr)

    include_optional = args.include
    if "all" in include_optional:
        include_optional = [flag for _, flag in OPTIONAL_TESTERS]

    generated_code = generate_test_file(model_info, args.model_filepath, include_optional, imports)

    if args.dry_run:
        print(generated_code)
    else:
        output_path = args.output or get_test_output_path(args.model_filepath)
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            f.write(generated_code)

        print(f"Generated test file: {output_path}")
        print(f"Model class: {model_info['name']}")
        print(f"Detected attributes: {list(model_info['attributes'].keys())}")


if __name__ == "__main__":
    main()
