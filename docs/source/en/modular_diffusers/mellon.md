<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->


## Using Custom Blocks with Mellon

[Mellon](https://github.com/cubiq/Mellon) is a visual workflow interface that integrates with Modular Diffusers and is designed for node-based workflows.

> [!WARNING]
> Mellon is in early development and not ready for production use yet. Consider this a sneak peek of how the integration works!


Custom blocks work in Mellon out of the box - just need to add a `mellon_pipeline_config.json` to your repository. This config file tells Mellon how to render your block's parameters as UI components.

Here's what it looks like in action with the [Gemini Prompt Expander](https://huggingface.co/diffusers/gemini-prompt-expander-mellon) block:

![Mellon custom block demo](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/modular_demo_dynamic.gif)

To use a modular diffusers custom block in Mellon:
1. Drag a **Dynamic Block Node** from the ModularDiffusers section
2. Enter the `repo_id` (e.g., `diffusers/gemini-prompt-expander-mellon`)
3. Click **Load Custom Block**
4. The node transforms to show your block's inputs and outputs

Now let's walk through how to create this config for your own custom block.

## Steps to create a Mellon config

1. **Specify Mellon types for your parameters** - Each `InputParam`/`OutputParam` needs a type that tells Mellon what UI component to render (e.g., `"textbox"`, `"dropdown"`, `"image"`).
2. **Generate `mellon_pipeline_config.json`** - Use our utility to generate a config template and push it to your Hub repository.
3. **(Optional) Manually adjust the config** - Fine-tune the generated config for your specific needs.

## Specify Mellon types for parameters

Mellon types determine how each parameter renders in the UI. If you don't specify a type for a parameter, it will default to `"custom"`, which renders as a simple connection dot. You can always adjust this later in the generated config.


| Type | Input/Output | Description |
|------|--------------|-------------|
| `image` | Both | Image (PIL Image) |
| `video` | Both | Video |
| `text` | Both | Text display |
| `textbox` | Input | Text input |
| `dropdown` | Input | Dropdown selection menu |
| `slider` | Input | Slider for numeric values |
| `number` | Input | Numeric input |
| `checkbox` | Input | Boolean toggle |

For parameters that need more configuration (like dropdowns with options, or sliders with min/max values), pass a `MellonParam` instance directly instead of a string. You can use one of the class methods below, or create a fully custom one with `MellonParam(name, label, type, ...)`.

| Method | Description |
|--------|-------------|
| `MellonParam.Input.image(name)` | Image input |
| `MellonParam.Input.textbox(name, default)` | Text input as textarea |
| `MellonParam.Input.dropdown(name, options, default)` | Dropdown selection |
| `MellonParam.Input.slider(name, default, min, max, step)` | Slider for numeric values |
| `MellonParam.Input.number(name, default, min, max, step)` | Numeric input (no slider) |
| `MellonParam.Input.seed(name, default)` | Seed input with randomize button |
| `MellonParam.Input.checkbox(name, default)` | Boolean checkbox |
| `MellonParam.Input.model(name)` | Model input for diffusers components |
| `MellonParam.Output.image(name)` | Image output |
| `MellonParam.Output.video(name)` | Video output |
| `MellonParam.Output.text(name)` | Text output |
| `MellonParam.Output.model(name)` | Model output for diffusers components |

Choose one of the methods below to specify a Mellon type.

### Using `metadata` in block definitions

If you're defining a custom block from scratch, add `metadata={"mellon": "<type>"}` directly to your `InputParam` and `OutputParam` definitions. If you're editing an existing custom block from the Hub, see [Editing custom blocks](./custom_blocks#editing-custom-blocks) for how to download it locally.

```python
class GeminiPromptExpander(ModularPipelineBlocks):
    
    @property
    def inputs(self) -> List[InputParam]:
        return [
            InputParam(
                "prompt",
                type_hint=str,
                required=True,
                description="Prompt to use",
                metadata={"mellon": "textbox"},  # Text input
            )
        ]
    
    @property
    def intermediate_outputs(self) -> List[OutputParam]:
        return [
            OutputParam(
                "prompt",
                type_hint=str,
                description="Expanded prompt by the LLM",
                metadata={"mellon": "text"},  # Text output
            ),
            OutputParam(
                "old_prompt",
                type_hint=str,
                description="Old prompt provided by the user",
                # No metadata - we don't want to render this in UI
            )
        ]
```

For full control over UI configuration, pass a `MellonParam` instance directly:
```python
from diffusers.modular_pipelines.mellon_node_utils import MellonParam

InputParam(
    "mode",
    type_hint=str,
    default="balanced",
    metadata={"mellon": MellonParam.Input.dropdown("mode", options=["fast", "balanced", "quality"])},
)
```

### Using `input_types` and `output_types` when Generating Config

If you're working with an existing pipeline or prefer to keep your block definitions clean, specify types when generating the config using the `input_types/output_types` argument:
```python
from diffusers.modular_pipelines.mellon_node_utils import MellonPipelineConfig

mellon_config = MellonPipelineConfig.from_custom_block(
    blocks,
    input_types={"prompt": "textbox"},
    output_types={"prompt": "text"}
)
```

> [!NOTE]
> When both `metadata` and `input_types`/`output_types` are specified, the arguments overrides `metadata`.

## Generate and push the Mellon config

After adding metadata to your block, generate the default Mellon configuration template and push it to the Hub:

```python
from diffusers import ModularPipelineBlocks
from diffusers.modular_pipelines.mellon_node_utils import MellonPipelineConfig

# load your custom blocks from your local dir
blocks = ModularPipelineBlocks.from_pretrained("/path/local/folder", trust_remote_code=True)

# Generate the default config template
mellon_config = MellonPipelineConfig.from_custom_block(blocks)
# push the default template to `repo_id`, you will need to pass the same local folder path so that it will save the config locally first
mellon_config.save(
    local_dir="/path/local/folder",
    repo_id= repo_id,
    push_to_hub=True
)
```

This creates a `mellon_pipeline_config.json` file in your repository.

## Review and adjust the config

The generated template is a starting point - you may want to adjust it for your needs. Let's walk through the generated config for the Gemini Prompt Expander:

```json
{
  "label": "Gemini Prompt Expander",
  "default_repo": "",
  "default_dtype": "",
  "node_params": {
    "custom": {
      "params": {
        "prompt": {
          "label": "Prompt",
          "type": "string",
          "display": "textarea",
          "default": ""
        },
        "out_prompt": {
          "label": "Prompt",
          "type": "string",
          "display": "output"
        },
        "old_prompt": {
          "label": "Old Prompt",
          "type": "custom",
          "display": "output"
        },
        "doc": {
          "label": "Doc",
          "type": "string",
          "display": "output"
        }
      },
      "input_names": ["prompt"],
      "model_input_names": [],
      "output_names": ["out_prompt", "old_prompt", "doc"],
      "block_name": "custom",
      "node_type": "custom"
    }
  }
}
```

### Understanding the Structure

The `params` dict defines how each UI element renders. The `input_names`, `model_input_names`, and `output_names` lists map these UI elements to the underlying [`ModularPipelineBlocks`]'s I/O interface:

| Mellon Config | ModularPipelineBlocks |
|---------------|----------------------|
| `input_names` | `inputs` property |
| `model_input_names` | `expected_components` property |
| `output_names` | `intermediate_outputs` property |

In this example: `prompt` is the only input. There are no model components, and outputs include `out_prompt`, `old_prompt`, and `doc`.

Now let's look at the `params` dict:

- **`prompt`**: An input parameter with `display: "textarea"` (renders as a text input box), `label: "Prompt"` (shown in the UI), and `default: ""` (starts empty). The `type: "string"` field is important in Mellon because it determines which nodes can connect together - only matching types can be linked with "noodles".

- **`out_prompt`**: The expanded prompt output. The `out_` prefix was automatically added because the input and output share the same name (`prompt`), avoiding naming conflicts in the config. It has `display: "output"` which renders as an output socket.

- **`old_prompt`**: Has `type: "custom"` because we didn't specify metadata. This renders as a simple dot in the UI. Since we don't actually want to expose this in the UI, we can remove it.

- **`doc`**: The documentation output, automatically added to all custom blocks.

### Making Adjustments

Remove `old_prompt` from both `params` and `output_names` because you won't need to use it.

```json
{
  "label": "Gemini Prompt Expander",
  "default_repo": "",
  "default_dtype": "",
  "node_params": {
    "custom": {
      "params": {
        "prompt": {
          "label": "Prompt",
          "type": "string",
          "display": "textarea",
          "default": ""
        },
        "out_prompt": {
          "label": "Prompt",
          "type": "string",
          "display": "output"
        },
        "doc": {
          "label": "Doc",
          "type": "string",
          "display": "output"
        }
      },
      "input_names": ["prompt"],
      "model_input_names": [],
      "output_names": ["out_prompt", "doc"],
      "block_name": "custom",
      "node_type": "custom"
    }
  }
}
```

See the final config at [diffusers/gemini-prompt-expander-mellon](https://huggingface.co/diffusers/gemini-prompt-expander-mellon).