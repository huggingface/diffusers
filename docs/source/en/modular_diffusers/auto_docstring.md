<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Auto docstring and parameter templates

Every [`~modular_pipelines.ModularPipelineBlocks`] has a `doc` property that is automatically generated from its `description`, `inputs`, `intermediate_outputs`, `expected_components`, and `expected_configs`. The auto docstring system keeps docstrings in sync with the block's actual interface. Parameter templates provide standardized descriptions for parameters that appear across many pipelines.

## Auto docstring

Modular pipeline blocks are composable — you can nest them, chain them in sequences, and rearrange them freely. Their docstrings follow the same pattern. When a [`~modular_pipelines.SequentialPipelineBlocks`] aggregates inputs and outputs from its sub-blocks, the documentation should update automatically without manual rewrites.

The `# auto_docstring` marker generates docstrings from the block's properties. Add it above a class definition to mark the class for automatic docstring generation.

```py
# auto_docstring
class FluxTextEncoderStep(SequentialPipelineBlocks):
    ...
```

Run the following command to generate and insert the docstrings.

```bash
python utils/modular_auto_docstring.py --fix_and_overwrite
```

The utility reads the block's `doc` property and inserts it as the class docstring.

```py
# auto_docstring
class FluxTextEncoderStep(SequentialPipelineBlocks):
    """
    Text input processing step that standardizes text embeddings for the pipeline.

    Inputs:
        prompt_embeds (`torch.Tensor`) *required*:
            text embeddings used to guide the image generation.
        ...

    Outputs:
        prompt_embeds (`torch.Tensor`):
            text embeddings used to guide the image generation.
        ...
    """
```

You can also check without overwriting, or target a specific file or directory.

```bash
# Check that all marked classes have up-to-date docstrings
python utils/modular_auto_docstring.py

# Check a specific file or directory
python utils/modular_auto_docstring.py src/diffusers/modular_pipelines/flux/
```

If any marked class is missing a docstring, the check fails and lists the classes that need updating.

```
Found the following # auto_docstring markers that need docstrings:
- src/diffusers/modular_pipelines/flux/encoders.py: FluxTextEncoderStep at line 42

Run `python utils/modular_auto_docstring.py --fix_and_overwrite` to fix them.
```

## Parameter templates

`InputParam` and `OutputParam` define a block's inputs and outputs. Create them directly or use `.template()` for standardized definitions of common parameters like `prompt`, `num_inference_steps`, or `latents`.

### InputParam

[`~modular_pipelines.InputParam`] describes a single input to a block.

| Field | Type | Description |
|---|---|---|
| `name` | `str` | Name of the parameter |
| `type_hint` | `Any` | Type annotation (e.g., `str`, `torch.Tensor`) |
| `default` | `Any` | Default value (if not set, parameter has no default) |
| `required` | `bool` | Whether the parameter is required |
| `description` | `str` | Human-readable description |
| `kwargs_type` | `str` | Group name for related parameters (e.g., `"denoiser_input_fields"`) |
| `metadata` | `dict` | Arbitrary additional information |

#### Creating InputParam directly

```py
from diffusers.modular_pipelines import InputParam

InputParam(
    name="guidance_scale",
    type_hint=float,
    default=7.5,
    description="Scale for classifier-free guidance.",
)
```

#### Using a template

```py
InputParam.template("prompt")
# Equivalent to:
# InputParam(name="prompt", type_hint=str, required=True,
#            description="The prompt or prompts to guide image generation.")
```

Templates set `name`, `type_hint`, `default`, `required`, and `description` automatically. Override any field or add context with the `note` parameter.

```py
# Override the default value
InputParam.template("num_inference_steps", default=28)

# Add a note to the description
InputParam.template("prompt_embeds", note="batch-expanded")
# description becomes: "text embeddings used to guide the image generation. ... (batch-expanded)"
```

### OutputParam

[`~modular_pipelines.OutputParam`] describes a single output from a block.

| Field | Type | Description |
|---|---|---|
| `name` | `str` | Name of the parameter |
| `type_hint` | `Any` | Type annotation |
| `description` | `str` | Human-readable description |
| `kwargs_type` | `str` | Group name for related parameters |
| `metadata` | `dict` | Arbitrary additional information |

#### Creating OutputParam directly

```py
from diffusers.modular_pipelines import OutputParam

OutputParam(name="image_latents", type_hint=torch.Tensor, description="Encoded image latents.")
```

#### Using a template

```py
OutputParam.template("latents")

# Add a note to the description
OutputParam.template("prompt_embeds", note="batch-expanded")
```

## Available templates

`INPUT_PARAM_TEMPLATES` and `OUTPUT_PARAM_TEMPLATES` are defined in [modular_pipeline_utils.py](https://github.com/huggingface/diffusers/blob/main/src/diffusers/modular_pipelines/modular_pipeline_utils.py). They include common parameters like `prompt`, `image`, `num_inference_steps`, `latents`, `prompt_embeds`, and more. Refer to the source for the full list of available template names.

