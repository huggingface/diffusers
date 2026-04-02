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
from dataclasses import dataclass, field
from textwrap import dedent

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
    sub_block_names: list = field(default_factory=list)


def daggr_command_factory(args: Namespace):
    return DaggrCommand(
        repo_id=args.repo_id,
        output=args.output,
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
        daggr_parser.set_defaults(func=daggr_command_factory)

    def __init__(
        self,
        repo_id: str,
        output: str | None = None,
        workflow: str | None = None,
        trigger_inputs: list | None = None,
    ):
        self.repo_id = repo_id
        self.output = output
        self.workflow = workflow
        self.trigger_inputs = trigger_inputs

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

        block_infos = _analyze_blocks(exec_blocks)
        _filter_outputs(block_infos)
        _classify_inputs(block_infos)

        workflow_label = self.workflow or "default"

        if self.output:
            workflow_resolve_code = self._get_workflow_resolve_code()
            code = _generate_code(block_infos, self.repo_id, blocks_class_name, workflow_label, workflow_resolve_code)

            try:
                ast.parse(code)
            except SyntaxError as e:
                logger.warning(f"Generated code has syntax error: {e}")

            with open(self.output, "w") as f:
                f.write(code)

            print(f"Generated daggr app: {self.output}")
            print(f"  Pipeline: {blocks_class_name}")
            print(f"  Workflow: {workflow_label}")
            print(f"  Blocks: {len(block_infos)}")
            print(f"\nRun with: python {self.output}")
        else:
            print(f"Launching daggr app for {blocks_class_name} ({workflow_label} workflow)...")
            graph = _build_graph(block_infos, pipeline, exec_blocks, blocks_class_name, workflow_label)
            graph.launch()

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


def _collapse_block_infos(block_infos):
    from collections import OrderedDict

    groups = OrderedDict()
    for info in block_infos:
        prefix = info.name.split(".")[0] if "." in info.name else info.name
        groups.setdefault(prefix, []).append(info)

    collapsed = []
    for prefix, group in groups.items():
        if len(group) == 1:
            collapsed.append(group[0])
            continue

        external_inputs = []
        seen_input_names = set()
        prior_outputs = set()
        for info in group:
            for inp in info.inputs:
                if inp.name and inp.name not in prior_outputs and inp.name not in seen_input_names:
                    external_inputs.append(inp)
                    seen_input_names.add(inp.name)
            for out in info.outputs:
                if out.name:
                    prior_outputs.add(out.name)

        all_outputs = []
        seen_output_names = set()
        for info in group:
            for out in info.outputs:
                if out.name and out.name not in seen_output_names:
                    all_outputs.append(out)
                    seen_output_names.add(out.name)

        merged = BlockInfo(
            name=prefix,
            class_name=group[0].class_name,
            description=prefix.replace("_", " ").title(),
            inputs=external_inputs,
            outputs=all_outputs,
            sub_block_names=[info.name for info in group],
        )
        collapsed.append(merged)

    return collapsed


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
            if out.name in downstream_input_names or (is_last and not _is_internal_type(out.type_hint))
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
# Gradio component code strings
# ---------------------------------------------------------------------------


def _type_hint_to_gradio(type_hint, param_name, default=None):
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

    if type_hint is str:
        lines = 1
        default_repr = f", value={default!r}" if default is not None else ""
        return f'gr.Textbox(label="{param_name}", lines={lines}{default_repr})'

    if type_hint is int:
        val = f", value={default!r}" if default is not None else ""
        return f'gr.Number(label="{param_name}", precision=0{val})'

    if type_hint is float:
        val = f", value={default!r}" if default is not None else ""
        return f'gr.Number(label="{param_name}"{val})'

    if type_hint is bool:
        val = default if default is not None else False
        return f'gr.Checkbox(label="{param_name}", value={val!r})'

    if type_hint is not None:
        type_str = str(type_hint)
        if "Image" in type_str:
            if "list" in type_str.lower():
                return f'gr.Gallery(label="{param_name}")'
            return f'gr.Image(label="{param_name}")'

    if default is not None:
        return f'gr.Textbox(label="{param_name}", value={default!r})'

    return f'gr.Textbox(label="{param_name}")'


def _output_type_to_gradio(type_hint, param_name):
    type_str = str(type_hint) if type_hint is not None else ""
    if "Image" in type_str:
        if "list" in type_str.lower():
            return f'gr.Gallery(label="{param_name}")'
        return f'gr.Image(label="{param_name}")'
    if type_hint is str:
        return f'gr.Textbox(label="{param_name}")'
    if type_hint is int or type_hint is float:
        return f'gr.Number(label="{param_name}")'
    return f'gr.Textbox(label="{param_name}", interactive=False)'


# ---------------------------------------------------------------------------
# Code generation
# ---------------------------------------------------------------------------


def _generate_code(block_infos, repo_id, blocks_class_name, workflow_label, workflow_resolve_code):
    sections = []

    sections.append(
        dedent(f"""\
        \"""Daggr app for {blocks_class_name} ({workflow_label} workflow)
        Generated by: diffusers-cli daggr
        \"""

        import gradio as gr from daggr import FnNode, InputNode, Graph


        _pipeline = None _exec_blocks = None


        def _get_pipeline():
            global _pipeline, _exec_blocks if _pipeline is None:
                from diffusers import ModularPipeline

                _pipeline = ModularPipeline.from_pretrained({repo_id!r}, trust_remote_code=True)
                _pipeline.load_components() _exec_blocks = {workflow_resolve_code}
            return _pipeline, _exec_blocks
    """)
    )

    for info in block_infos:
        fn_name = f"run_{_sanitize_name(info.name)}"
        input_names = [inp.name for inp in info.inputs if inp.name is not None]
        params = ", ".join(input_names)

        set_lines = "\n".join(f'    state.set("{n}", {n})' for n in input_names)

        if info.sub_block_names:
            block_calls = "\n".join(
                f'    _, state = exec_blocks.sub_blocks["{n}"](pipe, state)' for n in info.sub_block_names
            )
        else:
            block_calls = f'    _, state = exec_blocks.sub_blocks["{info.name}"](pipe, state)'

        if len(info.outputs) == 0:
            return_line = "    return None"
        elif len(info.outputs) == 1:
            return_line = f'    return state.get("{info.outputs[0].name}")'
        else:
            out_exprs = ", ".join(f'"{o.name}": state.get("{o.name}")' for o in info.outputs)
            return_line = f"    return {{{out_exprs}}}"

        sections.append(
            dedent(f"""\

            def {fn_name}({params}):
                from diffusers.modular_pipelines.modular_pipeline import PipelineState

                pipe, exec_blocks = _get_pipeline() state = PipelineState()
            {set_lines} {block_calls} {return_line}
        """)
        )

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

    sections.append(
        dedent(f"""

        # -- Graph -- graph = Graph("{graph_name}", nodes=[{nodes_str}]) graph.launch()
    """)
    )

    return "\n".join(sections)


# ---------------------------------------------------------------------------
# Direct graph construction (default launch path)
# ---------------------------------------------------------------------------


def _create_gradio_component(type_hint, param_name, default=None):
    import gradio as gr

    if param_name in SLIDER_PARAMS:
        opts = SLIDER_PARAMS[param_name]
        val = default if default is not None else opts.get("minimum", 0)
        return gr.Slider(
            label=param_name, value=val, minimum=opts["minimum"], maximum=opts["maximum"], step=opts["step"]
        )

    if type_hint is not None and _is_internal_type(type_hint):
        return None

    if type_hint is str:
        lines = 1
        kwargs = {"label": param_name, "lines": lines}
        if default is not None:
            kwargs["value"] = default
        return gr.Textbox(**kwargs)

    if type_hint is int:
        kwargs = {"label": param_name, "precision": 0}
        if default is not None:
            kwargs["value"] = default
        return gr.Number(**kwargs)

    if type_hint is float:
        kwargs = {"label": param_name}
        if default is not None:
            kwargs["value"] = default
        return gr.Number(**kwargs)

    if type_hint is bool:
        val = default if default is not None else False
        return gr.Checkbox(label=param_name, value=val)

    if type_hint is not None:
        type_str = str(type_hint)
        if "Image" in type_str:
            if "list" in type_str.lower():
                return gr.Gallery(label=param_name)
            return gr.Image(label=param_name)

    if default is not None:
        return gr.Textbox(label=param_name, value=default)

    return gr.Textbox(label=param_name)


def _create_output_component(type_hint, param_name):
    import gradio as gr

    type_str = str(type_hint) if type_hint is not None else ""
    if "Image" in type_str:
        if "list" in type_str.lower():
            return gr.Gallery(label=param_name)
        return gr.Image(label=param_name)
    if type_hint is str:
        return gr.Textbox(label=param_name)
    if type_hint is int or type_hint is float:
        return gr.Number(label=param_name)
    return gr.Textbox(label=param_name, interactive=False)


def _build_graph(block_infos, pipeline, exec_blocks, blocks_class_name, workflow_label):
    import inspect

    from daggr import FnNode, Graph, InputNode

    from ..modular_pipelines.modular_pipeline import PipelineState

    _components_loaded = False

    def _ensure_components():
        nonlocal _components_loaded
        if not _components_loaded:
            pipeline.load_components()
            _components_loaded = True

    fn_nodes = {}
    input_nodes = []
    user_input_sources = {}

    for info in block_infos:
        block_outputs = info.outputs
        input_names = [inp.name for inp in info.inputs if inp.name is not None]

        if info.sub_block_names:
            sub_blocks = [exec_blocks.sub_blocks[n] for n in info.sub_block_names]
        else:
            sub_blocks = [exec_blocks.sub_blocks[info.name]]

        def _make_wrapper(blocks_to_run, outputs, param_names):
            def wrapper(**kwargs):
                _ensure_components()
                state = PipelineState()
                for k, v in kwargs.items():
                    state.set(k, v)
                for blk in blocks_to_run:
                    _, state = blk(pipeline, state)
                if len(outputs) == 0:
                    return None
                elif len(outputs) == 1:
                    return state.get(outputs[0].name)
                else:
                    return tuple(state.get(o.name) for o in outputs)

            params = [inspect.Parameter(n, inspect.Parameter.POSITIONAL_OR_KEYWORD) for n in param_names]
            wrapper.__signature__ = inspect.Signature(params)
            return wrapper

        wrapper_fn = _make_wrapper(sub_blocks, block_outputs, input_names)
        display_name = info.name.replace("_", " ").replace(".", " > ").title()

        new_user_inputs = [inp for inp in info.user_inputs if inp.name not in user_input_sources]
        for inp in new_user_inputs:
            resolved_type, resolved_default = _resolve_from_template(inp)
            comp = _create_gradio_component(resolved_type, inp.name, resolved_default)
            if comp is not None:
                input_node = InputNode(inp.name, ports={inp.name: comp})
                input_nodes.append(input_node)
                user_input_sources[inp.name] = input_node

        inputs_dict = {}
        for inp in info.inputs:
            if inp.name is None:
                continue
            connected = False
            for conn_name, source_block in info.port_connections:
                if conn_name == inp.name:
                    inputs_dict[inp.name] = getattr(fn_nodes[source_block], inp.name)
                    connected = True
                    break
            if not connected:
                if inp.name in user_input_sources:
                    inputs_dict[inp.name] = getattr(user_input_sources[inp.name], inp.name)
                elif inp.default is not None:
                    inputs_dict[inp.name] = inp.default
                else:
                    inputs_dict[inp.name] = None

        outputs_dict = {}
        for out in block_outputs:
            comp = _create_output_component(out.type_hint, out.name)
            outputs_dict[out.name] = comp

        fn_node = FnNode(
            fn=wrapper_fn,
            name=display_name,
            inputs=inputs_dict if inputs_dict else None,
            outputs=outputs_dict if outputs_dict else None,
        )
        fn_nodes[info.name] = fn_node

    all_nodes = input_nodes + [fn_nodes[info.name] for info in block_infos]
    graph_name = f"{blocks_class_name} - {workflow_label}"
    return Graph(graph_name, nodes=all_nodes)
