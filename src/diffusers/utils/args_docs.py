# coding=utf-8
# Copyright 2025 HuggingFace Inc. team. All rights reserved.
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
Adapted from
https://github.com/huggingface/transformers/blob/5a95ed5ca0826c867e35e52f698db4d8fc907bcb/src/transformers/utils/args_doc.py
"""

import inspect
import os
import textwrap
from pathlib import Path
from typing import Optional, Union, get_args

import regex as re

from .doc import PT_SAMPLE_DOCSTRINGS, _prepare_output_docstrings
from .generic import ModelOutput


PATH_TO_diffusers = Path("src").resolve() / "diffusers"


AUTODOC_FILES = [
    "pipeline_*.py"
]

_re_checkpoint = re.compile(r"\[(.+?)\]\((https://huggingface\.co/.+?)\)")

class PipelineArgs:
    prompt = {
        "description": """
The prompt or prompts to guide the image generation. If not defined, one has to
pass `prompt_embeds` instead.
"""
    }

    prompt_2 = {
        "description": """
The prompt or prompts to be sent to the `tokenizer_2` and `text_encoder_2`. If
not defined, `prompt` is used in both text-encoders.
"""
    }

    height = {
        "description": """
The height in pixels of the generated image. Defaults to **1024** for best
results. Anything below **512** pixels won’t work well for
[stabilityai/stable-diffusion-xl-base-1.0] and other checkpoints not fine-tuned
for low resolutions.
"""
    }

    width = {
        "description": """
The width in pixels of the generated image. Defaults to **1024** for best
results. Anything below **512** pixels won’t work well for
[stabilityai/stable-diffusion-xl-base-1.0] and other checkpoints not fine-tuned
for low resolutions.
"""
    }

    num_inference_steps = {
        "description": """
The number of denoising steps. More steps usually yield higher-quality images
at the cost of slower inference.
"""
    }

    timesteps = {
        "description": """
Custom timesteps for schedulers that accept a `timesteps` argument in
`set_timesteps`. Must be a descending list. If not provided, the scheduler
derives timesteps from `num_inference_steps`.
"""
    }

    sigmas = {
        "description": """
Custom sigmas for schedulers that accept a `sigmas` argument in
`set_timesteps`. If not provided, sigmas are derived from
`num_inference_steps`.
"""
    }

    denoising_end = {
        "description": """
Fraction (0.0 – 1.0) of the denoising process after which inference stops
early, leaving residual noise. Useful in “Mixture of Denoisers” setups (see
*Refining the Image Output* in the documentation).
"""
    }

    guidance_scale = {
        "description": """
Classifier-free guidance scale (**w** in Imagen Eq. 2). Set `> 1` to enforce
closer adherence to the text prompt; higher values often trade off overall
image quality.
"""
    }

    negative_prompt = {
        "description": """
Prompt(s) *not* to guide generation. If omitted, supply
`negative_prompt_embeds` instead. Ignored when `guidance_scale < 1`.
"""
    }

    negative_prompt_2 = {
        "description": """
Prompt(s) *not* to guide generation for `tokenizer_2` / `text_encoder_2`. Falls
back to `negative_prompt` if not provided.
"""
    }

    num_images_per_prompt = {
        "description": """
Number of images to generate per prompt.
"""
    }

    eta = {
        "description": """
DDIM **η** parameter (see the DDIM paper). Only affects `DDIMScheduler`;
ignored for other schedulers.
"""
    }

    generator = {
        "description": """
A `torch.Generator` (or list of generators) for deterministic sampling.
"""
    }

    latents = {
        "description": """
Pre-generated Gaussian-sampled latents to seed generation. Enables reproducible
runs with different text inputs. If omitted, latents are auto-sampled using the
given `generator`.
"""
    }

    prompt_embeds = {
        "description": """
Pre-computed text embeddings, useful for prompt weighting or other advanced
manipulations. Generated from `prompt` if not supplied.
"""
    }

    negative_prompt_embeds = {
        "description": """
Pre-computed negative text embeddings. Generated from `negative_prompt` if not
supplied.
"""
    }

    pooled_prompt_embeds = {
        "description": """
Pre-generated pooled text embeddings for prompt weighting. Generated from
`prompt` if not supplied.
"""
    }

    negative_pooled_prompt_embeds = {
        "description": """
Pre-generated negative pooled text embeddings. Generated from `negative_prompt`
if not supplied.
"""
    }

    ip_adapter_image = {
        "description": """
Optional image input when using IP-Adapter.
"""
    }

    ip_adapter_image_embeds = {
        "description": """
Pre-generated image embeddings for IP-Adapter. Provide a list matching the
number of adapters. Each tensor: `(batch_size, num_images, emb_dim)`. Include a
negative embedding if `do_classifier_free_guidance` is `True`. Generated from
`ip_adapter_image` if omitted.
"""
    }

    output_type = {
        "description": """
Return format for generated images: `"pil"` for `PIL.Image.Image` or `"np"` for
`numpy.ndarray`. Defaults to `"pil"`.
"""
    }

    return_dict = {
        "description": """
If `True`, return a `StableDiffusionXLPipelineOutput`; otherwise return a tuple.
"""
    }

    cross_attention_kwargs = {
        "description": """
Dictionary passed to the pipeline’s `AttentionProcessor` for advanced
cross-attention control.
"""
    }

    guidance_rescale = {
        "description": """
Guidance rescale factor **φ** (see *Common Diffusion Noise Schedules and Sample
Steps are Flawed*, Eq. 16). Mitigates over-exposure when using zero terminal
SNR.
"""
    }

    original_size = {
        "description": """
Original image size `(height, width)` for SDXL micro-conditioning (sec. 2.2).
Defaults to `(height, width)` if unspecified.
"""
    }

    crops_coords_top_left = {
        "description": """
Coordinates of the top-left corner for a virtual crop. `(0, 0)` usually yields
well-centered images. Part of SDXL micro-conditioning.
"""
    }

    target_size = {
        "description": """
Desired output image size `(height, width)`. Defaults to `(height, width)` if
unspecified. Part of SDXL micro-conditioning.
"""
    }

    negative_original_size = {
        "description": """
Negatively conditions generation on a particular image resolution (SDXL
micro-conditioning, sec. 2.2). See issue #4208 for details.
"""
    }

    negative_crops_coords_top_left = {
        "description": """
Negatively conditions generation on specific crop coordinates (SDXL
micro-conditioning, sec. 2.2). See issue #4208 for details.
"""
    }

    negative_target_size = {
        "description": """
Negatively conditions generation on a target resolution, typically matching
`target_size` (SDXL micro-conditioning, sec. 2.2). See issue #4208 for details.
"""
    }

    callback_on_step_end = {
        "description": """
Function or subclass of `PipelineCallback` / `MultiPipelineCallbacks` invoked at
the end of each denoising step:

`callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
callback_kwargs: Dict)`

`callback_kwargs` includes tensors specified via
`callback_on_step_end_tensor_inputs`.
"""
    }

    callback_on_step_end_tensor_inputs = {
        "description": """
List of tensor names forwarded to `callback_on_step_end` through
`callback_kwargs`. Only tensors declared in the pipeline’s
`._callback_tensor_inputs` are allowed.
"""
    }


class ClassDocstring:
    Text2ImagePipeline = r"""
    Pipeline for text-to-image generation using Stable Diffusion XL.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.).
    """


class ClassAttrs:
    # fmt: off
    model_cpu_offload_seq = r"""
    A string representation of the CPU offloading sequence of model-level components of the pipeline to follow.
    """
    _optional_components = r"""
    Name of the components that are optional to load the underlying pipeline.
    """
    _callback_tensor_inputs = r"""
    Name of the pipeline inputs that are allowed to be used with callbacks.
    """
    # fmt: on


ARGS_TO_IGNORE = {"self", "kwargs", "args", "deprecated_arguments"}


def get_indent_level(func):
    # Use this instead of `inspect.getsource(func)` as getsource can be very slow
    return (len(func.__qualname__.split(".")) - 1) * 4


def equalize_indent(docstring, indent_level):
    """
    Adjust the indentation of a docstring to match the specified indent level.
    """
    # fully dedent the docstring
    docstring = "\n".join([line.lstrip() for line in docstring.splitlines()])
    return textwrap.indent(docstring, " " * indent_level)


def set_min_indent(docstring, indent_level):
    """
    Adjust the indentation of a docstring to match the specified indent level.
    """
    return textwrap.indent(textwrap.dedent(docstring), " " * indent_level)


def parse_shape(docstring):
    shape_pattern = re.compile(r"(of shape\s*(?:`.*?`|\(.*?\)))")
    match = shape_pattern.search(docstring)
    if match:
        return " " + match.group(1)
    return None


def parse_default(docstring):
    default_pattern = re.compile(r"(defaults to \s*[^)]*)")
    match = default_pattern.search(docstring)
    if match:
        return " " + match.group(1)
    return None


def parse_docstring(docstring, max_indent_level=0):
    """
    Parse the docstring to extract the Args section and return it as a dictionary.
    The docstring is expected to be in the format:
    Args:
        arg1 (type): Description of arg1.
        arg2 (type): Description of arg2.

    # This function will also return the remaining part of the docstring after the Args section.
    Returns:/Example:
    ...
    """
    match = re.search(r"(?m)^([ \t]*)(?=Example|Return)", docstring)
    if match:
        remainder_docstring = docstring[match.start() :]
        docstring = docstring[: match.start()]
    else:
        remainder_docstring = ""
    args_pattern = re.compile(r"(?:Args:)(\n.*)?(\n)?$", re.DOTALL)

    args_match = args_pattern.search(docstring)
    # still try to find args description in the docstring, if args are not preceded by "Args:"
    args_section = args_match.group(1).lstrip("\n") if args_match else docstring
    if args_section.split("\n")[-1].strip() == '"""':
        args_section = "\n".join(args_section.split("\n")[:-1])
    if args_section.split("\n")[0].strip() == 'r"""' or args_section.split("\n")[0].strip() == '"""':
        args_section = "\n".join(args_section.split("\n")[1:])
    args_section = set_min_indent(args_section, 0)

    params = {}
    if args_section:
        param_pattern = re.compile(
            # |--- Group 1 ---|| Group 2 ||- Group 3 -||---------- Group 4 ----------|
            rf"^\s{{0,{max_indent_level}}}(\w+)\s*\(\s*([^, \)]*)(\s*.*?)\s*\)\s*:\s*((?:(?!\n^\s{{0,{max_indent_level}}}\w+\s*\().)*)",
            re.DOTALL | re.MULTILINE,
        )
        for match in param_pattern.finditer(args_section):
            param_name = match.group(1)
            param_type = match.group(2)
            # param_type = match.group(2).replace("`", "")
            additional_info = match.group(3)
            optional = "optional" in additional_info
            shape = parse_shape(additional_info)
            default = parse_default(additional_info)
            param_description = match.group(4).strip()
            # set first line of param_description to 4 spaces:
            param_description = re.sub(r"^", " " * 4, param_description, 1)
            param_description = f"\n{param_description}"
            params[param_name] = {
                "type": param_type,
                "description": param_description,
                "optional": optional,
                "shape": shape,
                "default": default,
                "additional_info": additional_info,
            }

    if params and remainder_docstring:
        remainder_docstring = "\n" + remainder_docstring

    remainder_docstring = set_min_indent(remainder_docstring, 0)

    return params, remainder_docstring


def contains_type(type_hint, target_type) -> tuple[bool, Optional[object]]:
    """
    Check if a "nested" type hint contains a specific target type,
    return the first-level type containing the target_type if found.
    """
    args = get_args(type_hint)
    if args == ():
        try:
            return issubclass(type_hint, target_type), type_hint
        except Exception as _:
            return issubclass(type(type_hint), target_type), type_hint
    found_type_tuple = [contains_type(arg, target_type)[0] for arg in args]
    found_type = any(found_type_tuple)
    if found_type:
        type_hint = args[found_type_tuple.index(True)]
    return found_type, type_hint


def get_model_name(obj):
    """
    Get the model name from the file path of the object.
    """
    path = inspect.getsourcefile(obj)
    if path.split(os.path.sep)[-3] != "models":
        return None
    file_name = path.split(os.path.sep)[-1]
    for file_type in AUTODOC_FILES:
        start = file_type.split("*")[0]
        end = file_type.split("*")[-1] if "*" in file_type else ""
        if file_name.startswith(start) and file_name.endswith(end):
            model_name_lowercase = file_name[len(start) : -len(end)]
            return model_name_lowercase
    else:
        print(f"🚨 Something went wrong trying to find the model name in the path: {path}")
        return "model"


def get_placeholders_dict(placeholders: list, model_name: str) -> dict:
    """
    Get the dictionary of placeholders for the given model name.
    """
    # import here to avoid circular import
    from transformers.models import auto as auto_module

    placeholders_dict = {}
    for placeholder in placeholders:
        # Infer placeholders from the model name and the auto modules
        if placeholder in PLACEHOLDER_TO_AUTO_MODULE:
            place_holder_value = getattr(
                getattr(auto_module, PLACEHOLDER_TO_AUTO_MODULE[placeholder][0]),
                PLACEHOLDER_TO_AUTO_MODULE[placeholder][1],
            )[model_name]
            if isinstance(place_holder_value, (list, tuple)):
                place_holder_value = place_holder_value[0]
            placeholders_dict[placeholder] = place_holder_value

    return placeholders_dict


def format_args_docstring(args, model_name):
    """
    Replaces placeholders such as {image_processor_class} in the docstring with the actual values,
    deducted from the model name and the auto modules.
    """
    # first check if there are any placeholders in the args, if not return them as is
    placeholders = set(re.findall(r"{(.*?)}", "".join((args[arg]["description"] for arg in args))))
    if not placeholders:
        return args

    # get the placeholders dictionary for the given model name
    placeholders_dict = get_placeholders_dict(placeholders, model_name)

    # replace the placeholders in the args with the values from the placeholders_dict
    for arg in args:
        new_arg = args[arg]["description"]
        placeholders = re.findall(r"{(.*?)}", new_arg)
        placeholders = [placeholder for placeholder in placeholders if placeholder in placeholders_dict]
        if placeholders:
            new_arg = new_arg.format(**{placeholder: placeholders_dict[placeholder] for placeholder in placeholders})
        args[arg]["description"] = new_arg

    return args


def source_args_doc(args_classes: Union[object, list[object]]) -> dict:
    if isinstance(args_classes, (list, tuple)):
        args_classes_dict = {}
        for args_class in args_classes:
            args_classes_dict.update(args_class.__dict__)
        return args_classes_dict
    return args_classes.__dict__


def get_checkpoint_from_config_class(model_class):
    checkpoint = None

    # source code of `config_class`
    # config_source = inspect.getsource(config_class)
    config_source = model_class.config.__doc__
    checkpoints = _re_checkpoint.findall(config_source)
    # Each `checkpoint` is a tuple of a checkpoint name and a checkpoint link.
    # For example, `('google-bert/bert-base-uncased', 'https://huggingface.co/google-bert/bert-base-uncased')`
    for ckpt_name, ckpt_link in checkpoints:
        # allow the link to end with `/`
        if ckpt_link.endswith("/"):
            ckpt_link = ckpt_link[:-1]

        # verify the checkpoint name corresponds to the checkpoint link
        ckpt_link_from_name = f"https://huggingface.co/{ckpt_name}"
        if ckpt_link == ckpt_link_from_name:
            checkpoint = ckpt_name
            break

    return checkpoint


def add_intro_docstring(func, class_name, parent_class=None, indent_level=0):
    intro_docstring = ""
    if func.__name__ == "__call__":
        intro_docstring = r"""
        Function invoked when calling the pipeline for generation.
        """
        intro_docstring = equalize_indent(intro_docstring, indent_level + 4)

    return intro_docstring


def _get_model_info(func, parent_class):
    """
    Extract model information from a function or its parent class.

    Args:
        func (`function`): The function to extract information from
        parent_class (`class`): Optional parent class of the function
    """
    # import here to avoid circular import
    # TODO: Implement this.

    # Get model name from either parent class or function
    if parent_class is not None:
        model_name_lowercase = get_model_name(parent_class)
    else:
        model_name_lowercase = get_model_name(func)

    # Get class name from function's qualified name
    class_name = func.__qualname__.split(".")[0]

    # Get config class for the model
    if model_name_lowercase is None:
        config_class = None
    else:
        raise NotImplementedError

    return model_name_lowercase, class_name, config_class


def _process_parameter_type(param, param_name, func):
    """
    Process and format a parameter's type annotation.

    Args:
        param (`inspect.Parameter`): The parameter from the function signature
        param_name (`str`): The name of the parameter
        func (`function`): The function the parameter belongs to
    """
    optional = False
    if param.annotation != inspect.Parameter.empty:
        param_type = param.annotation
        if "typing" in str(param_type):
            param_type = "".join(str(param_type).split("typing.")).replace("transformers.", "~")
        elif hasattr(param_type, "__module__"):
            param_type = f"{param_type.__module__.replace('transformers.', '~').replace('builtins', '')}.{param.annotation.__name__}"
            if param_type[0] == ".":
                param_type = param_type[1:]
        else:
            if False:
                print(
                    f"🚨 {param_type} for {param_name} of {func.__qualname__} in file {func.__code__.co_filename} has an invalid type"
                )
        if "ForwardRef" in param_type:
            param_type = re.sub(r"ForwardRef\('([\w.]+)'\)", r"\1", param_type)
        if "Optional" in param_type:
            param_type = re.sub(r"Optional\[(.*?)\]", r"\1", param_type)
            optional = True
    else:
        param_type = ""

    return param_type, optional


def _get_parameter_info(param_name, documented_params, source_args_dict, param_type, optional):
    """
    Get parameter documentation details from the appropriate source.
    Tensor shape, optional status and description are taken from the custom docstring in priority if available.
    Type is taken from the function signature first, then from the custom docstring if missing from the signature

    Args:
        param_name (`str`): Name of the parameter
        documented_params (`dict`): Dictionary of documented parameters (manually specified in the docstring)
        source_args_dict (`dict`): Default source args dictionary to use if not in documented_params
        param_type (`str`): Current parameter type (may be updated)
        optional (`bool`): Whether the parameter is optional (may be updated)
    """
    description = None
    shape = None
    shape_string = ""
    is_documented = True
    additional_info = None

    if param_name in documented_params:
        # Parameter is documented in the function's docstring
        if param_type == "" and documented_params[param_name].get("type", None) is not None:
            param_type = documented_params[param_name]["type"]
        optional = documented_params[param_name]["optional"]
        shape = documented_params[param_name]["shape"]
        shape_string = shape if shape else ""
        additional_info = documented_params[param_name]["additional_info"] or ""
        description = f"{documented_params[param_name]['description']}\n"
    elif param_name in source_args_dict:
        # Parameter is documented in ModelArgs or ImageProcessorArgs
        shape = source_args_dict[param_name]["shape"]
        shape_string = " " + shape if shape else ""
        description = source_args_dict[param_name]["description"]
        additional_info = None
    else:
        # Parameter is not documented
        is_documented = False
    optional_string = r", *optional*" if optional else ""

    return param_type, optional_string, shape_string, additional_info, description, is_documented


def _process_regular_parameters(sig, func, class_name, documented_params, indent_level, undocumented_parameters):
    """
    Process all regular parameters (not kwargs parameters) from the function signature.

    Args:
        sig (`inspect.Signature`): Function signature
        func (`function`): Function the parameters belong to
        class_name (`str`): Name of the class
        documented_params (`dict`): Dictionary of parameters that are already documented
        indent_level (`int`): Indentation level
        undocumented_parameters (`list`): List to append undocumented parameters to
    """
    docstring = ""
    source_args_dict = source_args_doc([ModelArgs, ImageProcessorArgs])
    missing_args = {}

    for param_name, param in sig.parameters.items():
        # Skip parameters that should be ignored
        if (
            param_name in ARGS_TO_IGNORE
            or param.kind == inspect.Parameter.VAR_POSITIONAL
            or param.kind == inspect.Parameter.VAR_KEYWORD
        ):
            continue

        # Process parameter type and optional status
        param_type, optional = _process_parameter_type(param, param_name, func)

        # Check for default value
        param_default = ""
        if param.default != inspect._empty and param.default is not None:
            param_default = f", defaults to `{str(param.default)}`"

        param_type, optional_string, shape_string, additional_info, description, is_documented = _get_parameter_info(
            param_name, documented_params, source_args_dict, param_type, optional
        )

        if is_documented:
            if param_name == "config":
                if param_type == "":
                    param_type = f"[`{class_name}`]"
                else:
                    param_type = f"[`{param_type.split('.')[-1]}`]"
            elif param_type == "" and False:  # TODO: Enforce typing for all parameters
                print(f"🚨 {param_name} for {func.__qualname__} in file {func.__code__.co_filename} has no type")
            param_type = param_type if "`" in param_type else f"`{param_type}`"
            # Format the parameter docstring
            if additional_info:
                param_docstring = f"{param_name} ({param_type}{additional_info}):{description}"
            else:
                param_docstring = (
                    f"{param_name} ({param_type}{shape_string}{optional_string}{param_default}):{description}"
                )
            docstring += set_min_indent(
                param_docstring,
                indent_level + 8,
            )
        else:
            missing_args[param_name] = {
                "type": param_type if param_type else "<fill_type>",
                "optional": optional,
                "shape": shape_string,
                "description": description if description else "\n    <fill_description>",
                "default": param_default,
            }
            undocumented_parameters.append(
                f"🚨 `{param_name}` is part of {func.__qualname__}'s signature, but not documented. Make sure to add it to the docstring of the function in {func.__code__.co_filename}."
            )

    return docstring, missing_args


def find_sig_line(lines, line_end):
    parenthesis_count = 0
    sig_line_end = line_end
    found_sig = False
    while not found_sig:
        for char in lines[sig_line_end]:
            if char == "(":
                parenthesis_count += 1
            elif char == ")":
                parenthesis_count -= 1
                if parenthesis_count == 0:
                    found_sig = True
                    break
        sig_line_end += 1
    return sig_line_end


def _process_kwargs_parameters(
    sig, func, parent_class, model_name_lowercase, documented_kwargs, indent_level, undocumented_parameters
):
    """
    Process **kwargs parameters if needed.

    Args:
        sig (`inspect.Signature`): Function signature
        func (`function`): Function the parameters belong to
        parent_class (`class`): Parent class of the function
        model_name_lowercase (`str`): Lowercase model name
        documented_kwargs (`dict`): Dictionary of kwargs that are already documented
        indent_level (`int`): Indentation level
        undocumented_parameters (`list`): List to append undocumented parameters to
    """
    docstring = ""
    source_args_dict = source_args_doc(ImageProcessorArgs)

    # Check if we need to add typed kwargs description to the docstring
    unroll_kwargs = func.__name__ in UNROLL_KWARGS_METHODS
    if not unroll_kwargs and parent_class is not None:
        # Check if the function has a parent class with unroll kwargs
        unroll_kwargs = any(
            unroll_kwargs_class in parent_class.__name__ for unroll_kwargs_class in UNROLL_KWARGS_CLASSES
        )

    if unroll_kwargs:
        # get all unpackable "kwargs" parameters
        kwargs_parameters = [
            kwargs_param
            for _, kwargs_param in sig.parameters.items()
            if kwargs_param.kind == inspect.Parameter.VAR_KEYWORD
        ]
        for kwarg_param in kwargs_parameters:
            # If kwargs not typed, skip
            if kwarg_param.annotation == inspect.Parameter.empty:
                continue

            # Extract documentation for kwargs
            kwargs_documentation = kwarg_param.annotation.__args__[0].__doc__
            if kwargs_documentation is not None:
                documented_kwargs, _ = parse_docstring(kwargs_documentation)
                if model_name_lowercase is not None:
                    documented_kwargs = format_args_docstring(documented_kwargs, model_name_lowercase)

            # Process each kwarg parameter
            for param_name, param_type_annotation in kwarg_param.annotation.__args__[0].__annotations__.items():
                param_type = str(param_type_annotation)
                optional = False

                # Process parameter type
                if "typing" in param_type:
                    param_type = "".join(param_type.split("typing.")).replace("transformers.", "~")
                else:
                    param_type = f"{param_type.replace('transformers.', '~').replace('builtins', '')}.{param_name}"
                if "ForwardRef" in param_type:
                    param_type = re.sub(r"ForwardRef\('([\w.]+)'\)", r"\1", param_type)
                if "Optional" in param_type:
                    param_type = re.sub(r"Optional\[(.*?)\]", r"\1", param_type)
                    optional = True

                # Check for default value
                param_default = ""
                if parent_class is not None:
                    param_default = str(getattr(parent_class, param_name, ""))
                    param_default = f", defaults to `{param_default}`" if param_default != "" else ""

                param_type, optional_string, shape_string, additional_info, description, is_documented = (
                    _get_parameter_info(param_name, documented_kwargs, source_args_dict, param_type, optional)
                )

                if is_documented:
                    # Check if type is missing
                    if param_type == "":
                        print(
                            f"🚨 {param_name} for {kwarg_param.annotation.__args__[0].__qualname__} in file {func.__code__.co_filename} has no type"
                        )
                    param_type = param_type if "`" in param_type else f"`{param_type}`"
                    # Format the parameter docstring
                    if additional_info:
                        docstring += set_min_indent(
                            f"{param_name} ({param_type}{additional_info}):{description}",
                            indent_level + 8,
                        )
                    else:
                        docstring += set_min_indent(
                            f"{param_name} ({param_type}{shape_string}{optional_string}{param_default}):{description}",
                            indent_level + 8,
                        )
                else:
                    undocumented_parameters.append(
                        f"🚨 `{param_name}` is part of {kwarg_param.annotation.__args__[0].__qualname__}, but not documented. Make sure to add it to the docstring of the function in {func.__code__.co_filename}."
                    )

    return docstring


def _process_parameters_section(
    func_documentation, sig, func, class_name, model_name_lowercase, parent_class, indent_level
):
    """
    Process the parameters section of the docstring.

    Args:
        func_documentation (`str`): Existing function documentation (manually specified in the docstring)
        sig (`inspect.Signature`): Function signature
        func (`function`): Function the parameters belong to
        class_name (`str`): Name of the class the function belongs to
        model_name_lowercase (`str`): Lowercase model name
        parent_class (`class`): Parent class of the function (if any)
        indent_level (`int`): Indentation level
    """
    # Start Args section
    docstring = set_min_indent("Args:\n", indent_level + 4)
    undocumented_parameters = []
    documented_params = {}
    documented_kwargs = {}

    # Parse existing docstring if available
    if func_documentation is not None:
        documented_params, func_documentation = parse_docstring(func_documentation)
        if model_name_lowercase is not None:
            documented_params = format_args_docstring(documented_params, model_name_lowercase)

    # Process regular parameters
    param_docstring, missing_args = _process_regular_parameters(
        sig, func, class_name, documented_params, indent_level, undocumented_parameters
    )
    docstring += param_docstring

    # Process **kwargs parameters if needed
    kwargs_docstring = _process_kwargs_parameters(
        sig, func, parent_class, model_name_lowercase, documented_kwargs, indent_level, undocumented_parameters
    )
    docstring += kwargs_docstring

    # Report undocumented parameters
    if len(undocumented_parameters) > 0:
        print("\n".join(undocumented_parameters))

    return docstring


def _process_returns_section(func_documentation, sig, config_class, indent_level):
    """
    Process the returns section of the docstring.

    Args:
        func_documentation (`str`): Existing function documentation (manually specified in the docstring)
        sig (`inspect.Signature`): Function signature
        config_class (`str`): Config class for the model
        indent_level (`int`): Indentation level
    """
    return_docstring = ""

    # Extract returns section from existing docstring if available
    if (
        func_documentation is not None
        and (match_start := re.search(r"(?m)^([ \t]*)(?=Return)", func_documentation)) is not None
    ):
        match_end = re.search(r"(?m)^([ \t]*)(?=Example)", func_documentation)
        if match_end:
            return_docstring = func_documentation[match_start.start() : match_end.start()]
            func_documentation = func_documentation[match_end.start() :]
        else:
            return_docstring = func_documentation[match_start.start() :]
            func_documentation = ""
        return_docstring = set_min_indent(return_docstring, indent_level + 4)
    # Otherwise, generate return docstring from return annotation if available
    elif sig.return_annotation is not None and sig.return_annotation != inspect._empty:
        add_intro, return_annotation = contains_type(sig.return_annotation, ModelOutput)
        return_docstring = _prepare_output_docstrings(return_annotation, config_class, add_intro=add_intro)
        return_docstring = return_docstring.replace("typing.", "")
        return_docstring = set_min_indent(return_docstring, indent_level + 4)

    return return_docstring, func_documentation


def _process_example_section(
    func_documentation, func, parent_class, class_name, model_name_lowercase, config_class, checkpoint, indent_level
):
    """
    Process the example section of the docstring.

    Args:
        func_documentation (`str`): Existing function documentation (manually specified in the docstring)
        func (`function`): Function being processed
        parent_class (`class`): Parent class of the function
        class_name (`str`): Name of the class
        model_name_lowercase (`str`): Lowercase model name
        config_class (`str`): Config class for the model
        checkpoint: Checkpoint to use in examples
        indent_level (`int`): Indentation level
    """
    # Import here to avoid circular import
    from transformers.models import auto as auto_module

    example_docstring = ""

    # Use existing example section if available

    if func_documentation is not None and (match := re.search(r"(?m)^([ \t]*)(?=Example)", func_documentation)):
        example_docstring = func_documentation[match.start() :]
        example_docstring = "\n" + set_min_indent(example_docstring, indent_level + 4)
    # No examples for __init__ methods or if the class is not a model
    elif parent_class is None and model_name_lowercase is not None:
        task = rf"({'|'.join(PT_SAMPLE_DOCSTRINGS.keys())})"
        model_task = re.search(task, class_name)
        CONFIG_MAPPING = auto_module.configuration_auto.CONFIG_MAPPING

        # Get checkpoint example
        if (checkpoint_example := checkpoint) is None:
            try:
                checkpoint_example = get_checkpoint_from_config_class(CONFIG_MAPPING[model_name_lowercase])
            except KeyError:
                raise

        # Add example based on model task
        if model_task is not None:
            if checkpoint_example is not None:
                example_annotation = ""
                task = model_task.group()
                example_annotation = PT_SAMPLE_DOCSTRINGS[task].format(
                    checkpoint=checkpoint_example,
                )
                example_docstring = set_min_indent(example_annotation, indent_level + 4)
            else:
                print(
                    f"🚨 No checkpoint found for {class_name}.{func.__name__}. Please add a `checkpoint` arg to `auto_docstring` or add one in {config_class}'s docstring"
                )
        else:
            raise NotImplementedError

    return example_docstring


def auto_method_docstring(func, parent_class=None, custom_intro=None, custom_args=None, checkpoint=None):
    """
    Wrapper that automatically generates docstring.
    """

    # Use inspect to retrieve the method's signature
    sig = inspect.signature(func)
    indent_level = get_indent_level(func)

    # Get model information
    model_name_lowercase, class_name, config_class = _get_model_info(func, parent_class)
    func_documentation = func.__doc__
    if custom_args is not None and func_documentation is not None:
        func_documentation = set_min_indent(custom_args, indent_level + 4) + "\n" + func_documentation
    elif custom_args is not None:
        func_documentation = custom_args

    # Add intro to the docstring before args description if needed
    if custom_intro is not None:
        docstring = set_min_indent(custom_intro, indent_level + 4)
    else:
        docstring = add_intro_docstring(
            func, class_name=class_name, parent_class=parent_class, indent_level=indent_level
        )

    # Process Parameters section
    docstring += _process_parameters_section(
        func_documentation, sig, func, class_name, model_name_lowercase, parent_class, indent_level
    )

    # Process Returns section
    return_docstring, func_documentation = _process_returns_section(
        func_documentation, sig, config_class, indent_level
    )
    docstring += return_docstring

    # Process Example section
    example_docstring = _process_example_section(
        func_documentation,
        func,
        parent_class,
        class_name,
        model_name_lowercase,
        config_class,
        checkpoint,
        indent_level,
    )
    docstring += example_docstring

    # Assign the dynamically generated docstring to the wrapper function
    func.__doc__ = docstring
    return func


def auto_class_docstring(cls, custom_intro=None, custom_args=None, checkpoint=None):
    """
    Wrapper that automatically generates a docstring for classes based on their attributes and methods.
    """
    # import here to avoid circular import

    docstring_init = auto_method_docstring(cls.__init__, parent_class=cls, custom_args=custom_args).__doc__.replace(
        "Args:", "Parameters:"
    )
    indent_level = get_indent_level(cls)
    model_name_lowercase = get_model_name(cls)
    model_name_title = " ".join([k.title() for k in model_name_lowercase.split("_")]) if model_name_lowercase else None

    name = re.findall(rf"({'|'.join(ClassDocstring.__dict__.keys())})$", cls.__name__)
    if name == [] and cls.__doc__ is None and custom_intro is None:
        raise ValueError(
            f"`{cls.__name__}` is not part of the auto doc. Here are the available classes: {ClassDocstring.__dict__.keys()}"
        )
    if name != [] or custom_intro is not None:
        name = name[0] if name else None
        if custom_intro is not None:
            pre_block = equalize_indent(custom_intro, indent_level)
            if not pre_block.endswith("\n"):
                pre_block += "\n"
        elif model_name_title is None:
            pre_block = ""
        else:
            pre_block = getattr(ClassDocstring, name).format(model_name=model_name_title)
        # Start building the docstring
        docstring = set_min_indent(f"{pre_block}", indent_level) if len(pre_block) else ""
        if name != "PreTrainedModel" and "PreTrainedModel" in (x.__name__ for x in cls.__mro__):
            docstring += set_min_indent(f"{ClassDocstring.PreTrainedModel}", indent_level)
        # Add the __init__ docstring
        docstring += set_min_indent(f"\n{docstring_init}", indent_level)
        attr_docs = ""
        # Get all attributes and methods of the class

        for attr_name, attr_value in cls.__dict__.items():
            if not callable(attr_value) and not attr_name.startswith("__"):
                if attr_value.__class__.__name__ == "property":
                    attr_type = "property"
                else:
                    attr_type = type(attr_value).__name__
                if name and "Config" in name:
                    raise ValueError("Config should have explicit docstring")
                indented_doc = getattr(ClassAttrs, attr_name, None)
                if indented_doc is not None:
                    attr_docs += set_min_indent(f"{attr_name} (`{attr_type}`): {indented_doc}", 0)

        # TODO: Add support for Attributes section in docs
        # if len(attr_docs.replace(" ", "")):
        #     docstring += set_min_indent("\nAttributes:\n", indent_level)
        #     docstring += set_min_indent(attr_docs, indent_level + 4)
    else:
        print(
            f"You used `@auto_class_docstring` decorator on `{cls.__name__}` but this class is not part of the AutoMappings. Remove the decorator"
        )
    # Assign the dynamically generated docstring to the wrapper class
    cls.__doc__ = docstring

    return cls


def auto_docstring(obj=None, *, custom_intro=None, custom_args=None, checkpoint=None):
    """
    Automatically generates docstrings for classes and methods in the Diffusers library.

    This decorator can be used in the following forms:
    @auto_docstring
    def my_function(...):
        ...
    or
    @auto_docstring()
    def my_function(...):
        ...
    or
    @auto_docstring(custom_intro="Custom intro", ...)
    def my_function(...):
        ...

    Args:
        custom_intro (str, optional): Custom introduction text to add to the docstring. This will replace the default
            introduction text generated by the decorator before the Args section.
        checkpoint (str, optional): Checkpoint name to use in the docstring. This should be automatically inferred from the
            model configuration class, but can be overridden if needed.
    """

    def auto_docstring_decorator(obj):
        if len(obj.__qualname__.split(".")) > 1:
            return auto_method_docstring(
                obj, custom_args=custom_args, custom_intro=custom_intro, checkpoint=checkpoint
            )
        else:
            return auto_class_docstring(obj, custom_args=custom_args, custom_intro=custom_intro, checkpoint=checkpoint)

    if obj:
        return auto_docstring_decorator(obj)

    return auto_docstring_decorator
