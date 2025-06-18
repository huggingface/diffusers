
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
"""
Doc utilities: Utilities related to documentation

Adapted from:
https://github.com/huggingface/transformers/blob/5a95ed5ca0826c867e35e52f698db4d8fc907bcb/src/transformers/utils/doc.py
"""

import functools
import inspect
import re
import textwrap
import types
from collections import OrderedDict

from ..pipelines.auto_pipeline import AUTO_TEXT2IMAGE_PIPELINES_MAPPING


def get_docstring_indentation_level(func):
    """Return the indentation level of the start of the docstring of a class or function (or method)."""
    # We assume classes are always defined in the global scope
    if inspect.isclass(func):
        return 4
    source = inspect.getsource(func)
    first_line = source.splitlines()[0]
    function_def_level = len(first_line) - len(first_line.lstrip())
    return 4 + function_def_level


def add_start_docstrings(*docstr):
    def docstring_decorator(fn):
        fn.__doc__ = "".join(docstr) + (fn.__doc__ if fn.__doc__ is not None else "")
        return fn

    return docstring_decorator


def add_start_docstrings_to_model_forward(*docstr):
    def docstring_decorator(fn):
        class_name = f"[`{fn.__qualname__.split('.')[0]}`]"
        intro = rf"""    The {class_name} forward method, overrides the `__call__` special method.

    <Tip>

    Although the recipe for forward pass needs to be defined within this function, one should call the [`Module`]
    instance afterwards instead of this since the former takes care of running the pre and post processing steps while
    the latter silently ignores them.

    </Tip>
"""

        correct_indentation = get_docstring_indentation_level(fn)
        current_doc = fn.__doc__ if fn.__doc__ is not None else ""
        try:
            first_non_empty = next(line for line in current_doc.splitlines() if line.strip() != "")
            doc_indentation = len(first_non_empty) - len(first_non_empty.lstrip())
        except StopIteration:
            doc_indentation = correct_indentation

        docs = docstr
        # In this case, the correct indentation level (class method, 2 Python levels) was respected, and we should
        # correctly reindent everything. Otherwise, the doc uses a single indentation level
        if doc_indentation == 4 + correct_indentation:
            docs = [textwrap.indent(textwrap.dedent(doc), " " * correct_indentation) for doc in docstr]
            intro = textwrap.indent(textwrap.dedent(intro), " " * correct_indentation)

        docstring = "".join(docs) + current_doc
        fn.__doc__ = intro + docstring
        return fn

    return docstring_decorator


def add_end_docstrings(*docstr):
    def docstring_decorator(fn):
        fn.__doc__ = (fn.__doc__ if fn.__doc__ is not None else "") + "".join(docstr)
        return fn

    return docstring_decorator


PT_RETURN_INTRODUCTION = r"""
    Returns:
        [`{full_output_type}`] or `tuple(torch.FloatTensor)`: A [`{full_output_type}`] or a tuple of
        `torch.FloatTensor` (if `return_dict=False` is passed) comprising various
        elements depending on the model and inputs.

"""

TEXT_TO_IMAGE_PIPELINE_CLASSES = list({p[0] for p in AUTO_TEXT2IMAGE_PIPELINES_MAPPING})

def _get_indent(t):
    """Returns the indentation in the first line of t"""
    search = re.search(r"^(\s*)\S", t)
    return "" if search is None else search.groups()[0]


def _convert_output_args_doc(output_args_doc):
    """Convert output_args_doc to display properly."""
    # Split output_arg_doc in blocks argument/description
    indent = _get_indent(output_args_doc)
    blocks = []
    current_block = ""
    for line in output_args_doc.split("\n"):
        # If the indent is the same as the beginning, the line is the name of new arg.
        if _get_indent(line) == indent:
            if len(current_block) > 0:
                blocks.append(current_block[:-1])
            current_block = f"{line}\n"
        else:
            # Otherwise it's part of the description of the current arg.
            # We need to remove 2 spaces to the indentation.
            current_block += f"{line[2:]}\n"
    blocks.append(current_block[:-1])

    # Format each block for proper rendering
    for i in range(len(blocks)):
        blocks[i] = re.sub(r"^(\s+)(\S+)(\s+)", r"\1- **\2**\3", blocks[i])
        blocks[i] = re.sub(r":\s*\n\s*(\S)", r" -- \1", blocks[i])

    return "\n".join(blocks)


def _prepare_output_docstrings(output_type, config_class, min_indent=None, add_intro=True):
    """
    Prepares the return part of the docstring using `output_type`.
    """
    output_docstring = output_type.__doc__
    params_docstring = None
    if output_docstring is not None:
        # Remove the head of the docstring to keep the list of args only
        lines = output_docstring.split("\n")
        i = 0
        while i < len(lines) and re.search(r"^\s*(Args|Parameters):\s*$", lines[i]) is None:
            i += 1
        if i < len(lines):
            params_docstring = "\n".join(lines[(i + 1) :])
            params_docstring = _convert_output_args_doc(params_docstring)
        elif add_intro:
            raise ValueError(
                f"No `Args` or `Parameters` section is found in the docstring of `{output_type.__name__}`. Make sure it has "
                "docstring and contain either `Args` or `Parameters`."
            )

    # Add the return introduction
    if add_intro:
        full_output_type = f"{output_type.__module__}.{output_type.__name__}"
        intro = PT_RETURN_INTRODUCTION
        intro = intro.format(full_output_type=full_output_type, config_class=config_class)
    else:
        full_output_type = str(output_type)
        intro = f"\nReturns:\n    `{full_output_type}`"
        if params_docstring is not None:
            intro += ":\n"

    result = intro
    if params_docstring is not None:
        result += params_docstring

    # Apply minimum indent if necessary
    if min_indent is not None:
        lines = result.split("\n")
        # Find the indent of the first nonempty line
        i = 0
        while len(lines[i]) == 0:
            i += 1
        indent = len(_get_indent(lines[i]))
        # If too small, add indentation to all nonempty lines
        if indent < min_indent:
            to_add = " " * (min_indent - indent)
            lines = [(f"{to_add}{line}" if len(line) > 0 else line) for line in lines]
            result = "\n".join(lines)

    return result


FAKE_MODEL_DISCLAIMER = """
    <Tip warning={true}>

    This example uses a random model as the real ones are all very big. To get proper results, you should use
    {real_checkpoint} instead of {fake_checkpoint}. If you get out-of-memory when loading that checkpoint, you can
    refer to our optimization docs.

    </Tip>
"""


PT_TEXT_TO_IMAGE_SAMPLE = r"""
    Example:

    ```python
    >>> from diffusers import DiffusionPipeline
    >>> import torch

    >>> # If memory doesn't allow, enable optimizations like `enable_model_cpu_offload()`.
    >>> pipe = DiffusionPipeline.from_pretrained("{checkpoint}", torch_dtype=torch.bfloat16).to("cuda")

    >>> prompt = "a photo of a cute dog."
    >>> image = pipe(prompt).images[0] # Configure other pipe call arguments as needed.
    ```
"""

PT_SAMPLE_DOCSTRINGS = {
    "Text2Image": PT_TEXT_TO_IMAGE_SAMPLE
}
PIPELINE_TASKS_TO_SAMPLE_DOCSTRINGS = OrderedDict(["text-to-image", PT_TEXT_TO_IMAGE_SAMPLE])

def filter_outputs_from_example(docstring, **kwargs):
    """
    Removes the lines testing an output with the doctest syntax in a code sample when it's set to `None`.
    """
    for key, value in kwargs.items():
        if value is not None:
            continue

        doc_key = "{" + key + "}"
        docstring = re.sub(rf"\n([^\n]+)\n\s+{doc_key}\n", "\n", docstring)

    return docstring


def add_code_sample_docstrings(
    *docstr,
    checkpoint=None,
    output_type=None,
    config_class=None,
    model_cls=None,
):
    def docstring_decorator(fn):
        # model_class defaults to function's class if not specified otherwise
        model_class = fn.__qualname__.split(".")[0] if model_cls is None else model_cls

        sample_docstrings = PT_SAMPLE_DOCSTRINGS

        # putting all kwargs for docstrings in a dict to be used
        # with the `.format(**doc_kwargs)`. Note that string might
        # be formatted with non-existing keys, which is fine.
        doc_kwargs = {
            "checkpoint": checkpoint,
            "true": "{true}",  # For <Tip warning={true}> syntax that conflicts with formatting.
        }

        if model_class in TEXT_TO_IMAGE_PIPELINE_CLASSES:
            code_sample = sample_docstrings["Text2Image"]
        else:
            raise ValueError(f"Docstring can't be built for model {model_class}")

        code_sample = filter_outputs_from_example(code_sample)
        func_doc = (fn.__doc__ or "") + "".join(docstr)
        output_doc = "" if output_type is None else _prepare_output_docstrings(output_type, config_class)
        built_doc = code_sample.format(**doc_kwargs)

        fn.__doc__ = func_doc + output_doc + built_doc
        return fn

    return docstring_decorator


def replace_return_docstrings(output_type=None, config_class=None):
    def docstring_decorator(fn):
        func_doc = fn.__doc__
        lines = func_doc.split("\n")
        i = 0
        while i < len(lines) and re.search(r"^\s*Returns?:\s*$", lines[i]) is None:
            i += 1
        if i < len(lines):
            indent = len(_get_indent(lines[i]))
            lines[i] = _prepare_output_docstrings(output_type, config_class, min_indent=indent)
            func_doc = "\n".join(lines)
        else:
            raise ValueError(
                f"The function {fn} should have an empty 'Return:' or 'Returns:' in its docstring as placeholder, "
                f"current docstring is:\n{func_doc}"
            )
        fn.__doc__ = func_doc
        return fn

    return docstring_decorator


def copy_func(f):
    """Returns a copy of a function f."""
    # Based on http://stackoverflow.com/a/6528148/190597 (Glenn Maynard)
    g = types.FunctionType(f.__code__, f.__globals__, name=f.__name__, argdefs=f.__defaults__, closure=f.__closure__)
    g = functools.update_wrapper(g, f)
    g.__kwdefaults__ = f.__kwdefaults__
    return g
