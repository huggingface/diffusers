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
import tempfile
import types
import typing
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass, field

from ..utils import logging
from . import BaseDiffusersCLICommand


logger = logging.get_logger("diffusers-cli/daggr")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

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

DROPDOWN_PARAMS = {
    "output_type": {"choices": ["pil", "np", "pt", "latent"], "value": "pil"},
}

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


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
    sub_block_names: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# CLI command
# ---------------------------------------------------------------------------


def daggr_command_factory(args: Namespace):
    return DaggrCommand(
        repo_id=args.repo_id,
        output=args.output,
        workflow=getattr(args, "workflow", None),
        trigger_inputs=getattr(args, "trigger_inputs", None),
        deploy=getattr(args, "deploy", None),
        hardware=getattr(args, "hardware", "cpu-basic"),
        private=getattr(args, "private", False),
        requirements=getattr(args, "requirements", None),
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
            default=None,
            help="Save the generated app to a file instead of launching it directly.",
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
        daggr_parser.add_argument(
            "--deploy",
            type=str,
            default=None,
            metavar="SPACE_NAME",
            help="Deploy the generated app to a HuggingFace Space via daggr deploy.",
        )
        daggr_parser.add_argument(
            "--hardware",
            type=str,
            default="cpu-basic",
            help="Hardware tier for the deployed Space (default: cpu-basic). E.g. a10g-small, a100-large.",
        )
        daggr_parser.add_argument(
            "--private",
            action="store_true",
            default=False,
            help="Make the deployed Space private.",
        )
        daggr_parser.add_argument(
            "--requirements",
            type=str,
            default=None,
            help="Path to a requirements.txt file for the deployed Space.",
        )
        daggr_parser.set_defaults(func=daggr_command_factory)

    def __init__(
        self,
        repo_id: str,
        output: str | None = None,
        workflow: str | None = None,
        trigger_inputs: list | None = None,
        deploy: str | None = None,
        hardware: str = "cpu-basic",
        private: bool = False,
        requirements: str | None = None,
    ):
        self.repo_id = repo_id
        self.output = output
        self.workflow = workflow
        self.trigger_inputs = trigger_inputs
        self.deploy = deploy
        self.hardware = hardware
        self.private = private
        self.requirements = requirements

    def run(self):
        from ..modular_pipelines.modular_pipeline import ModularPipeline

        logger.info(f"Loading blocks from {self.repo_id}...")
        pipeline = ModularPipeline.from_pretrained(self.repo_id, trust_remote_code=True)
        blocks = pipeline._blocks
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

        block_infos = _get_block_info(exec_blocks)
        _filter_outputs(block_infos)
        _classify_inputs(block_infos)

        workflow_label = self.workflow or "default"
        workflow_resolve_code = self._get_workflow_resolve_code()

        code = _generate_code(block_infos, self.repo_id, blocks_class_name, workflow_label, workflow_resolve_code)

        try:
            ast.parse(code)
        except SyntaxError as e:
            logger.warning(f"Generated code has syntax error: {e}")

        if self.deploy:
            self._deploy_to_space(code)
        elif self.output:
            with open(self.output, "w") as f:
                f.write(code)

            print(f"Generated daggr app: {self.output}")
            print(f"  Pipeline: {blocks_class_name}")
            print(f"  Workflow: {workflow_label}")
            print(f"  Blocks: {len(block_infos)}")
            print(f"\nRun with: python {self.output}")
        else:
            print(f"Launching daggr app for {blocks_class_name} ({workflow_label} workflow)...")
            tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, prefix="daggr_")
            tmp.write(code)
            tmp.close()
            logger.info(f"Generated temp script: {tmp.name}")
            exec(compile(code, tmp.name, "exec"), {"__name__": "__main__"})

    def _deploy_to_space(self, code):
        import os
        from pathlib import Path

        from daggr.cli import _deploy

        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, prefix="daggr_")
        tmp.write(code)
        tmp.close()

        req_path = self.requirements
        if not req_path:
            requirements = "diffusers[torch]\ntransformers\naccelerate\nsentencepiece\nbitsandbytes\ndaggr\ngradio\n"
            req_file = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, prefix="daggr_req_")
            req_file.write(requirements)
            req_file.close()
            req_path = req_file.name

        secrets = {}
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            secrets["HF_TOKEN"] = hf_token

        _deploy(
            script_path=Path(tmp.name),
            name=self.deploy,
            title=None,
            org=None,
            private=self.private,
            hardware=self.hardware,
            secrets=secrets,
            requirements_path=req_path,
            dry_run=False,
        )

    def _get_workflow_resolve_code(self):
        if self.workflow:
            return f"_pipeline._blocks.get_workflow({self.workflow!r})"
        elif self.trigger_inputs:
            kwargs_str = ", ".join(f"{name!r}: True" for name in self.trigger_inputs)
            return f"_pipeline._blocks.get_execution_blocks(**{{{kwargs_str}}})"
        else:
            return "_pipeline._blocks.get_execution_blocks()"


# ---------------------------------------------------------------------------
# Block analysis
# ---------------------------------------------------------------------------


def _get_block_info(exec_blocks):
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


def _filter_outputs(block_infos):
    last_idx = len(block_infos) - 1
    for i, info in enumerate(block_infos):
        downstream_input_names = set()
        for later_info in block_infos[i + 1 :]:
            for inp in later_info.inputs:
                if inp.name:
                    downstream_input_names.add(inp.name)

        is_last = i == last_idx
        info.outputs = [
            out
            for out in info.outputs
            if out.name in downstream_input_names
            or (is_last and (_contains_pil_image(out.type_hint) or not _is_internal_type(out.type_hint)))
        ]


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
            elif _is_user_facing(inp):
                user_inputs.append(inp)
            else:
                fixed_inputs.append(inp)

        info.user_inputs = user_inputs
        info.port_connections = port_connections
        info.fixed_inputs = fixed_inputs

        for out in info.outputs:
            if out.name:
                all_prior_outputs[out.name] = info.name

    for info in block_infos:
        if not info.port_connections and not info.user_inputs:
            output_names = [o.name for o in info.outputs]
            consumed = any(
                o_name in {c[0] for other in block_infos for c in other.port_connections} for o_name in output_names
            )
            if not consumed and output_names:
                logger.warning(
                    f"Block '{info.name}' appears disconnected: "
                    f"outputs {output_names} are not consumed by any downstream block. "
                    f"inputs={[i.name for i in info.inputs]}"
                )


# ---------------------------------------------------------------------------
# Type helpers
# ---------------------------------------------------------------------------


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
        return False
    type_name = _get_type_name(type_hint)
    if type_name is None:
        return False
    if type_name in INTERNAL_TYPE_NAMES or type_name in INTERNAL_TYPE_FULL_NAMES:
        return True
    type_str = str(type_hint)
    for full_name in INTERNAL_TYPE_FULL_NAMES:
        if full_name in type_str:
            return True
    if type_str.startswith("dict[") or type_str == "dict":
        return True
    return False


def _contains_pil_image(type_hint):
    from PIL import Image

    if type_hint is Image.Image:
        return True
    args = typing.get_args(type_hint)
    return any(_contains_pil_image(a) for a in args) if args else False


def _is_list_or_tuple_type(type_hint):
    origin = typing.get_origin(type_hint)
    if origin in (list, tuple):
        return True
    if origin is typing.Union or origin is types.UnionType:
        return any(_is_list_or_tuple_type(a) for a in typing.get_args(type_hint))
    return False


def _resolve_from_template(inp):
    from ..modular_pipelines.modular_pipeline_utils import INPUT_PARAM_TEMPLATES

    type_hint = inp.type_hint
    default = inp.default
    if inp.name in INPUT_PARAM_TEMPLATES:
        tmpl = INPUT_PARAM_TEMPLATES[inp.name]
        if type_hint is None:
            type_hint = tmpl.get("type_hint")
        if default is None:
            default = tmpl.get("default")
    return type_hint, default


def _is_user_facing(inp):
    type_hint, _ = _resolve_from_template(inp)
    if type_hint is not None:
        if _contains_pil_image(type_hint):
            return True
        return not _is_internal_type(type_hint)
    if inp.name in SLIDER_PARAMS:
        return True

    return False


def _sanitize_name(name):
    sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    if sanitized and sanitized[0].isdigit():
        sanitized = f"_{sanitized}"
    return sanitized


# ---------------------------------------------------------------------------
# Gradio component code strings (for code generation)
# ---------------------------------------------------------------------------


def _type_hint_to_gradio(type_hint, param_name, default=None):
    if param_name in DROPDOWN_PARAMS:
        opts = DROPDOWN_PARAMS[param_name]
        val = default if default is not None else opts.get("value")
        return f'gr.Dropdown(label="{param_name}", choices={opts["choices"]!r}, value={val!r})'

    if param_name in SLIDER_PARAMS:
        opts = SLIDER_PARAMS[param_name]
        val = default if default is not None else opts.get("minimum", 0)
        return (
            f'gr.Slider(label="{param_name}", value={val!r}, '
            f"minimum={opts['minimum']}, maximum={opts['maximum']}, "
            f"step={opts['step']})"
        )

    if type_hint is not None and _is_internal_type(type_hint):
        return None

    if type_hint is not None and _contains_pil_image(type_hint):
        return f'gr.Image(label="{param_name}")'

    if type_hint is str:
        default_repr = f", value={default!r}" if default is not None else ""
        return f'gr.Textbox(label="{param_name}", lines=1{default_repr})'

    if type_hint is int:
        val = f", value={default!r}" if default is not None else ""
        return f'gr.Number(label="{param_name}", precision=0{val})'

    if type_hint is float:
        val = f", value={default!r}" if default is not None else ""
        return f'gr.Number(label="{param_name}"{val})'

    if type_hint is bool:
        val = default if default is not None else False
        return f'gr.Checkbox(label="{param_name}", value={val!r})'

    if default is not None:
        return f'gr.Textbox(label="{param_name}", value={default!r})'

    return f'gr.Textbox(label="{param_name}")'


def _output_type_to_gradio(type_hint, param_name):
    if type_hint is not None and _contains_pil_image(type_hint):
        return f'gr.Image(label="{param_name}")'
    if type_hint is str:
        return f'gr.Textbox(label="{param_name}")'
    if type_hint is int or type_hint is float:
        return f'gr.Number(label="{param_name}")'
    return f'gr.Textbox(label="{param_name}", visible=False)'


# ---------------------------------------------------------------------------
# Code generation
# ---------------------------------------------------------------------------


def _generate_code(block_infos, repo_id, blocks_class_name, workflow_label, workflow_resolve_code):
    sections = []

    # Collect blocks that need image pre/postprocessing
    blocks_with_image_inputs = {}
    blocks_with_parseable_inputs = {}
    blocks_with_image_outputs = set()

    for info in block_infos:
        img_names = set()
        parse_names = set()
        for inp in info.inputs:
            if inp.name is None:
                continue
            resolved_type, _ = _resolve_from_template(inp)
            if resolved_type is not None and _contains_pil_image(resolved_type):
                img_names.add(inp.name)
            elif resolved_type is not None and _is_list_or_tuple_type(resolved_type):
                parse_names.add(inp.name)
        if img_names:
            blocks_with_image_inputs[info.name] = img_names
        if parse_names:
            blocks_with_parseable_inputs[info.name] = parse_names
        if any(out.type_hint is not None and _contains_pil_image(out.type_hint) for out in info.outputs):
            blocks_with_image_outputs.add(info.name)

    needs_image_io = blocks_with_image_inputs or blocks_with_image_outputs
    needs_parsing = bool(blocks_with_parseable_inputs)

    # Header
    extra_imports = ""
    if needs_parsing:
        extra_imports += "import ast\n"
    if needs_image_io:
        extra_imports += "import tempfile\n"
    if extra_imports:
        extra_imports += "\n"

    sections.append(
        f'"""Daggr app for {blocks_class_name} ({workflow_label} workflow)\n'
        f"Generated by: diffusers-cli daggr\n"
        f'"""\n'
        f"\n"
        f"import os\n"
        f"{extra_imports}"
        f"import gradio as gr\n"
        f"from daggr import FnNode, InputNode, Graph\n"
        f"\n"
        f"\n"
        f"_pipeline = None\n"
        f"_exec_blocks = None\n"
        f"_state = None\n"
        f"\n"
        f"\n"
        f"def _get_pipeline():\n"
        f"    global _pipeline, _exec_blocks, _state\n"
        f"    if _pipeline is None:\n"
        f"        from diffusers import ModularPipeline\n"
        f"        from diffusers.modular_pipelines.modular_pipeline import PipelineState\n"
        f"\n"
        f"        import torch\n"
        f"\n"
        f'        _token = os.environ.get("HF_TOKEN")\n'
        f'        _device = "cuda" if torch.cuda.is_available() else "cpu"\n'
        f"        _pipeline = ModularPipeline.from_pretrained({repo_id!r}, trust_remote_code=True, token=_token)\n"
        f"        _pipeline.load_components(torch_dtype=torch.bfloat16, device_map=_device)\n"
        f"        _exec_blocks = {workflow_resolve_code}\n"
        f"        _state = PipelineState()\n"
        f"    return _pipeline, _exec_blocks, _state\n"
    )

    # Pre/postprocess helpers (only if needed)
    if needs_image_io:
        sections.append(
            "\ndef _save_image(val):\n"
            "    from PIL import Image as PILImage\n"
            "\n"
            "    if isinstance(val, list):\n"
            "        paths = [_save_image(item) for item in val]\n"
            "        return paths[0] if paths else None\n"
            "    if isinstance(val, PILImage.Image):\n"
            '        f = tempfile.NamedTemporaryFile(suffix=".png", delete=False)\n'
            "        val.save(f.name)\n"
            "        return f.name\n"
            "    return val\n"
        )

    # Block functions
    for info in block_infos:
        fn_name = f"run_{_sanitize_name(info.name)}"
        input_names = [inp.name for inp in info.inputs if inp.name is not None]
        params = ", ".join(input_names)

        body_lines = []
        body_lines.append("    pipe, exec_blocks, state = _get_pipeline()")

        # Preprocess user inputs
        user_input_names = {inp.name for inp in info.user_inputs}

        if info.name in blocks_with_image_inputs:
            body_lines.append("    from PIL import Image as PILImage")
            body_lines.append("")
            for img_name in blocks_with_image_inputs[info.name]:
                body_lines.append(f"    if {img_name} is not None and isinstance({img_name}, str):")
                body_lines.append(f"        {img_name} = PILImage.open({img_name})")

        if info.name in blocks_with_parseable_inputs:
            for parse_name in blocks_with_parseable_inputs[info.name]:
                body_lines.append(f"    if {parse_name} is not None and isinstance({parse_name}, str):")
                body_lines.append(
                    f"        {parse_name} = ast.literal_eval({parse_name}.strip()) if {parse_name}.strip() else None"
                )

        # Only set user-provided inputs into shared state (port connections already in state)
        for n in input_names:
            if n in user_input_names:
                body_lines.append(f'    state.set("{n}", {n})')

        if info.sub_block_names:
            for n in info.sub_block_names:
                body_lines.append(f'    _, state = exec_blocks.sub_blocks["{n}"](pipe, state)')
        else:
            body_lines.append(f'    _, state = exec_blocks.sub_blocks["{info.name}"](pipe, state)')

        has_image_out = info.name in blocks_with_image_outputs

        # Return serializable values for daggr; real tensor data stays in shared state
        if len(info.outputs) == 0:
            body_lines.append("    return None")
        elif len(info.outputs) == 1:
            out = info.outputs[0]
            if has_image_out and _contains_pil_image(out.type_hint):
                body_lines.append(f'    return _save_image(state.get("{out.name}"))')
            elif out.type_hint is not None and _is_internal_type(out.type_hint):
                body_lines.append(f'    return "{out.name}"')
            else:
                body_lines.append(f'    return state.get("{out.name}")')
        else:
            ret_exprs = []
            for o in info.outputs:
                if has_image_out and o.type_hint is not None and _contains_pil_image(o.type_hint):
                    ret_exprs.append(f'_save_image(state.get("{o.name}"))')
                elif o.type_hint is not None and _is_internal_type(o.type_hint):
                    ret_exprs.append(f'"{o.name}"')
                else:
                    ret_exprs.append(f'state.get("{o.name}")')
            body_lines.append(f"    return {', '.join(ret_exprs)}")

        body = "\n".join(body_lines)
        sections.append(f"\n\ndef {fn_name}({params}):\n{body}\n")

    # Pipeline blocks
    sections.append("\n# -- Pipeline Blocks --")

    node_var_names = {}
    input_node_var_names = []
    user_input_sources = {}

    for info in block_infos:
        var_name = f"{_sanitize_name(info.name)}_node"
        node_var_names[info.name] = var_name
        fn_name = f"run_{_sanitize_name(info.name)}"
        display_name = info.name.replace("_", " ").replace(".", " > ").title()

        new_user_inputs = [inp for inp in info.user_inputs if inp.name not in user_input_sources]
        for inp in new_user_inputs:
            resolved_type, resolved_default = _resolve_from_template(inp)
            gradio_comp = _type_hint_to_gradio(resolved_type, inp.name, resolved_default)
            if gradio_comp:
                input_var = f"{_sanitize_name(inp.name)}_input"
                sections.append(
                    f'\n{input_var} = InputNode("{inp.name}", ports={{\n    "{inp.name}": {gradio_comp},\n}})'
                )
                input_node_var_names.append(input_var)
                user_input_sources[inp.name] = input_var

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
                if inp.name in user_input_sources:
                    src = user_input_sources[inp.name]
                    input_entries.append(f'    "{inp.name}": {src}.{inp.name},')
                elif inp.default is not None:
                    input_entries.append(f'    "{inp.name}": {inp.default!r},')
                else:
                    input_entries.append(f'    "{inp.name}": None,')

        output_entries = []
        for out in info.outputs:
            gradio_out = _output_type_to_gradio(out.type_hint, out.name)
            output_entries.append(f'    "{out.name}": {gradio_out},')

        node_parts = [f"{var_name} = FnNode(", f"    fn={fn_name},", f'    name="{display_name}",']
        if input_entries:
            node_parts.append("    inputs={")
            node_parts.extend(input_entries)
            node_parts.append("    },")
        if output_entries:
            node_parts.append("    outputs={")
            node_parts.extend(output_entries)
            node_parts.append("    },")
        node_parts.append(")")

        sections.append("\n" + "\n".join(node_parts))

    all_node_vars = input_node_var_names + [node_var_names[info.name] for info in block_infos]
    graph_name = f"{blocks_class_name} - {workflow_label}"
    nodes_str = ", ".join(all_node_vars)

    sections.append(f'\n\n# -- Graph --\ngraph = Graph("{graph_name}", nodes=[{nodes_str}])\ngraph.launch()\n')

    return "\n".join(sections)
