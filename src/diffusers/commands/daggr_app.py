# Copyright 2025 The HuggingFace Team. All rights reserved.
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

import ast
import re
from argparse import ArgumentParser, Namespace
from collections import OrderedDict
from dataclasses import dataclass, field

from ..utils import logging
from . import BaseDiffusersCLICommand


logger = logging.get_logger("diffusers-cli/daggr")

INTERNAL_TYPE_NAMES = {
    "Tensor",
    "Generator",
}

INTERNAL_TYPE_FULL_NAMES = {
    "torch.Tensor",
    "torch.Generator",
    "torch.dtype",
}

SLIDER_PARAMS = {
    "height": {"minimum": 256, "maximum": 2048, "step": 64},
    "width": {"minimum": 256, "maximum": 2048, "step": 64},
    "num_inference_steps": {"minimum": 1, "maximum": 100, "step": 1},
    "guidance_scale": {"minimum": 0, "maximum": 30, "step": 0.5},
    "strength": {"minimum": 0, "maximum": 1, "step": 0.05},
    "control_guidance_start": {"minimum": 0, "maximum": 1, "step": 0.05},
    "control_guidance_end": {"minimum": 0, "maximum": 1, "step": 0.05},
    "controlnet_conditioning_scale": {"minimum": 0, "maximum": 2, "step": 0.1},
}


@dataclass
class BlockInfo:
    name: str
    class_name: str
    description: str
    inputs: list
    outputs: list
    user_inputs: list = field(default_factory=list)
    port_connections: list = field(default_factory=list)
    fixed_inputs: list = field(default_factory=list)


def daggr_command_factory(args: Namespace):
    return DaggrCommand(
        repo_id=args.repo_id,
        output=args.output or "daggr_app.py",
        workflow=getattr(args, "workflow", None),
        trigger_inputs=getattr(args, "trigger_inputs", None),
    )


class DaggrCommand(BaseDiffusersCLICommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        daggr_parser = parser.add_parser("daggr", help="Generate a daggr app from a modular pipeline repo.")
        daggr_parser.add_argument(
            "repo_id",
            type=str,
            help="HuggingFace Hub repo ID containing a modular pipeline (with modular_model_index.json).",
        )
        daggr_parser.add_argument(
            "--output",
            type=str,
            default="daggr_app.py",
            help="Output file path for the generated daggr app. Default: daggr_app.py",
        )
        daggr_parser.add_argument(
            "--workflow",
            type=str,
            default=None,
            help="Named workflow to resolve conditional blocks (e.g. 'text2image', 'image2image').",
        )
        daggr_parser.add_argument(
            "--trigger-inputs",
            nargs="*",
            default=None,
            help="Trigger input names for manual conditional resolution.",
        )
        daggr_parser.set_defaults(func=daggr_command_factory)

    def __init__(
        self,
        repo_id: str,
        output: str = "daggr_app.py",
        workflow: str | None = None,
        trigger_inputs: list | None = None,
    ):
        self.repo_id = repo_id
        self.output = output
        self.workflow = workflow
        self.trigger_inputs = trigger_inputs

    def run(self):
        from ..modular_pipelines.modular_pipeline import ModularPipelineBlocks

        logger.info(f"Loading blocks from {self.repo_id}...")
        blocks = ModularPipelineBlocks.from_pretrained(self.repo_id, trust_remote_code=True)
        blocks_class_name = blocks.__class__.__name__

        if self.workflow:
            logger.info(f"Resolving workflow: {self.workflow}")
            exec_blocks = blocks.get_workflow(self.workflow)
        elif self.trigger_inputs:
            trigger_kwargs = {name: True for name in self.trigger_inputs}
            logger.info(f"Resolving with trigger inputs: {self.trigger_inputs}")
            exec_blocks = blocks.get_execution_blocks(**trigger_kwargs)
        else:
            logger.info("Resolving default execution blocks...")
            exec_blocks = blocks.get_execution_blocks()

        block_infos = _analyze_blocks(exec_blocks)
        _classify_inputs(block_infos)

        workflow_label = self.workflow or "default"
        workflow_resolve_code = self._get_workflow_resolve_code()
        code = _generate_code(block_infos, self.repo_id, blocks_class_name, workflow_label, workflow_resolve_code)

        try:
            ast.parse(code)
        except SyntaxError as e:
            logger.warning(f"Generated code has syntax error: {e}")

        with open(self.output, "w") as f:
            f.write(code)

        logger.info(f"Daggr app written to {self.output}")
        print(f"Generated daggr app: {self.output}")
        print(f"  Pipeline: {blocks_class_name}")
        print(f"  Workflow: {workflow_label}")
        print(f"  Blocks: {len(block_infos)}")
        print(f"\nRun with: python {self.output}")

    def _get_workflow_resolve_code(self):
        if self.workflow:
            return f"_pipeline._blocks.get_workflow({self.workflow!r})"
        elif self.trigger_inputs:
            kwargs_str = ", ".join(f"{name!r}: True" for name in self.trigger_inputs)
            return f"_pipeline._blocks.get_execution_blocks(**{{{kwargs_str}}})"
        else:
            return "_pipeline._blocks.get_execution_blocks()"


def _analyze_blocks(exec_blocks):
    block_infos = []
    for name, block in exec_blocks.sub_blocks.items():
        info = BlockInfo(
            name=name,
            class_name=block.__class__.__name__,
            description=getattr(block, "description", "") or "",
            inputs=list(block.inputs) if hasattr(block, "inputs") else [],
            outputs=list(block.intermediate_outputs) if hasattr(block, "intermediate_outputs") else [],
        )
        block_infos.append(info)
    return block_infos


def _get_type_name(type_hint):
    if type_hint is None:
        return None
    if hasattr(type_hint, "__name__"):
        return type_hint.__name__
    if hasattr(type_hint, "__module__") and hasattr(type_hint, "__qualname__"):
        return f"{type_hint.__module__}.{type_hint.__qualname__}"
    return str(type_hint)


def _is_internal_type(type_hint):
    if type_hint is None:
        return True
    type_name = _get_type_name(type_hint)
    if type_name is None:
        return True
    if type_name in INTERNAL_TYPE_NAMES or type_name in INTERNAL_TYPE_FULL_NAMES:
        return True
    type_str = str(type_hint)
    for full_name in INTERNAL_TYPE_FULL_NAMES:
        if full_name in type_str:
            return True
    if type_str.startswith("dict[") or type_str == "dict":
        return True
    return False


def _type_hint_to_gradio(type_hint, param_name, default=None):
    if _is_internal_type(type_hint):
        return None

    if param_name in SLIDER_PARAMS:
        slider_opts = SLIDER_PARAMS[param_name]
        val = default if default is not None else slider_opts.get("minimum", 0)
        return (
            f'gr.Slider(label="{param_name}", value={val!r}, '
            f"minimum={slider_opts['minimum']}, maximum={slider_opts['maximum']}, "
            f"step={slider_opts['step']})"
        )

    type_name = _get_type_name(type_hint)
    type_str = str(type_hint)

    if type_name == "str" or type_hint is str:
        lines = 3 if "prompt" in param_name else 1
        default_repr = f", value={default!r}" if default is not None else ""
        return f'gr.Textbox(label="{param_name}", lines={lines}{default_repr})'

    if type_name == "int" or type_hint is int:
        val = f", value={default!r}" if default is not None else ""
        return f'gr.Number(label="{param_name}", precision=0{val})'

    if type_name == "float" or type_hint is float:
        val = f", value={default!r}" if default is not None else ""
        return f'gr.Number(label="{param_name}"{val})'

    if type_name == "bool" or type_hint is bool:
        val = default if default is not None else False
        return f'gr.Checkbox(label="{param_name}", value={val!r})'

    if "Image" in type_str:
        if "list" in type_str.lower():
            return f'gr.Gallery(label="{param_name}")'
        return f'gr.Image(label="{param_name}")'

    if default is not None:
        return f'gr.Textbox(label="{param_name}", value={default!r})'

    return f'gr.Textbox(label="{param_name}")'


def _output_type_to_gradio(type_hint, param_name):
    if _is_internal_type(type_hint):
        return None
    type_str = str(type_hint)
    if "Image" in type_str:
        if "list" in type_str.lower():
            return f'gr.Gallery(label="{param_name}")'
        return f'gr.Image(label="{param_name}")'
    if type_hint is str:
        return f'gr.Textbox(label="{param_name}")'
    if type_hint is int or type_hint is float:
        return f'gr.Number(label="{param_name}")'
    return None


def _classify_inputs(block_infos):
    all_prior_outputs = {}

    for info in block_infos:
        user_inputs = []
        port_connections = []
        fixed_inputs = []

        for inp in info.inputs:
            if inp.name is None:
                continue
            if inp.name in all_prior_outputs:
                port_connections.append((inp.name, all_prior_outputs[inp.name]))
            elif _is_internal_type(inp.type_hint):
                fixed_inputs.append(inp)
            else:
                user_inputs.append(inp)

        info.user_inputs = user_inputs
        info.port_connections = port_connections
        info.fixed_inputs = fixed_inputs

        for out in info.outputs:
            if out.name and out.name not in all_prior_outputs:
                all_prior_outputs[out.name] = info.name


def _sanitize_name(name):
    sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    if sanitized and sanitized[0].isdigit():
        sanitized = f"_{sanitized}"
    return sanitized


def _generate_code(block_infos, repo_id, blocks_class_name, workflow_label, workflow_resolve_code):
    lines = []

    lines.append(f'"""Daggr app for {blocks_class_name} ({workflow_label} workflow)')
    lines.append("Generated by: diffusers-cli daggr")
    lines.append('"""')
    lines.append("")
    lines.append("import gradio as gr")
    lines.append("from daggr import FnNode, InputNode, Graph")
    lines.append("")
    lines.append("")

    # Pipeline and resolved blocks loader
    lines.append("_pipeline = None")
    lines.append("_exec_blocks = None")
    lines.append("")
    lines.append("")
    lines.append("def _get_pipeline():")
    lines.append("    global _pipeline, _exec_blocks")
    lines.append("    if _pipeline is None:")
    lines.append("        from diffusers import ModularPipeline")
    lines.append(f"        _pipeline = ModularPipeline.from_pretrained({repo_id!r}, trust_remote_code=True)")
    lines.append("        _pipeline.load_components()")
    lines.append(f"        _exec_blocks = {workflow_resolve_code}")
    lines.append("    return _pipeline, _exec_blocks")
    lines.append("")
    lines.append("")

    # Wrapper functions
    for info in block_infos:
        fn_name = f"run_{_sanitize_name(info.name)}"
        all_input_names = []
        for inp in info.inputs:
            if inp.name is not None:
                all_input_names.append(inp.name)

        params = ", ".join(all_input_names)
        lines.append(f"def {fn_name}({params}):")
        lines.append("    from diffusers.modular_pipelines.modular_pipeline import PipelineState")
        lines.append("")
        lines.append("    pipe, exec_blocks = _get_pipeline()")
        lines.append("    state = PipelineState()")
        for inp_name in all_input_names:
            lines.append(f'    state.set("{inp_name}", {inp_name})')
        lines.append(f'    block = exec_blocks.sub_blocks["{info.name}"]')
        lines.append("    _, state = block(pipe, state)")

        if len(info.outputs) == 0:
            lines.append("    return None")
        elif len(info.outputs) == 1:
            out = info.outputs[0]
            lines.append(f'    return state.get("{out.name}")')
        else:
            out_names = [out.name for out in info.outputs]
            out_dict = ", ".join(f'"{n}": state.get("{n}")' for n in out_names)
            lines.append(f"    return {{{out_dict}}}")
        lines.append("")
        lines.append("")

    # Collect all user-facing inputs across blocks
    all_user_inputs = OrderedDict()
    for info in block_infos:
        for inp in info.user_inputs:
            if inp.name not in all_user_inputs:
                all_user_inputs[inp.name] = inp

    # InputNode
    if all_user_inputs:
        lines.append("# -- User Inputs --")
        lines.append('user_inputs = InputNode("User Inputs", ports={')
        for inp_name, inp in all_user_inputs.items():
            gradio_comp = _type_hint_to_gradio(inp.type_hint, inp_name, inp.default)
            if gradio_comp:
                lines.append(f'    "{inp_name}": {gradio_comp},')
        lines.append("})")
        lines.append("")
        lines.append("")

    # FnNode definitions
    lines.append("# -- Pipeline Blocks --")
    node_var_names = {}

    for info in block_infos:
        var_name = f"{_sanitize_name(info.name)}_node"
        node_var_names[info.name] = var_name
        fn_name = f"run_{_sanitize_name(info.name)}"

        display_name = info.name.replace("_", " ").replace(".", " > ").title()

        # Build inputs dict
        input_entries = []
        for inp in info.inputs:
            if inp.name is None:
                continue

            connected = False
            for conn_name, source_block in info.port_connections:
                if conn_name == inp.name:
                    source_var = node_var_names[source_block]
                    input_entries.append(f'    "{inp.name}": {source_var}.{inp.name},')
                    connected = True
                    break

            if not connected:
                if inp.name in all_user_inputs:
                    input_entries.append(f'    "{inp.name}": user_inputs.{inp.name},')
                elif inp.default is not None:
                    input_entries.append(f'    "{inp.name}": {inp.default!r},')
                else:
                    input_entries.append(f'    "{inp.name}": None,')

        # Build outputs dict
        output_entries = []
        for out in info.outputs:
            gradio_out = _output_type_to_gradio(out.type_hint, out.name)
            if gradio_out:
                output_entries.append(f'    "{out.name}": {gradio_out},')
            else:
                output_entries.append(f'    "{out.name}": None,')

        lines.append(f"{var_name} = FnNode(")
        lines.append(f"    fn={fn_name},")
        lines.append(f'    name="{display_name}",')

        if input_entries:
            lines.append("    inputs={")
            lines.extend(input_entries)
            lines.append("    },")

        if output_entries:
            lines.append("    outputs={")
            lines.extend(output_entries)
            lines.append("    },")

        lines.append(")")
        lines.append("")

    # Graph
    lines.append("")
    lines.append("# -- Graph --")
    all_node_vars = []
    if all_user_inputs:
        all_node_vars.append("user_inputs")
    all_node_vars.extend(node_var_names[info.name] for info in block_infos)

    graph_name = f"{blocks_class_name} - {workflow_label}"
    nodes_str = ", ".join(all_node_vars)
    lines.append(f'graph = Graph("{graph_name}", nodes=[{nodes_str}])')
    lines.append("graph.launch()")
    lines.append("")

    return "\n".join(lines)
