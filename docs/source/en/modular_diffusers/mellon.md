<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->


## Using Custom Blocks with Mellon

[Mellon](https://github.com/cubiq/Mellon) is a visual workflow interface (similar to ComfyUI) that integrates with Modular Diffusers. This guide shows how to add Mellon support to your custom blocks so they can be used in the Mellon UI.

## Overview

To use a custom block in Mellon, you need to:

1. **Add `metadata` field to your block's `InputParam` and `OutputParam` definitions** - This is a simple string that tells Mellon what UI component to use (e.g., `"textbox"`, `"dropdown"`, `"image"`)
2. **Generate `mellon_pipeline_config.json`** - We provide a utility to generate a default template and push it to your Hub repository
3. **(Optional) Manually adjust the template** - Adjust the generated config for your specific needs

## Step 1: Add Metadata to Input/Output Parameters

The `metadata` field tells Mellon how to render each parameter in the UI. Add `metadata={"mellon": "<type>"}` to your `InputParam` and `OutputParam` definitions.

> [!NOTE]
> If you don't specify metadata for a parameter, it will default to `"custom"` type, which renders as a simple connection dot in the UI. You can always adjust this later in the generated config.

### Supported Mellon Types

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

### Example: Adding Metadata

Here's an example using the [Gemini Prompt Expander](https://huggingface.co/diffusers-internal-dev/gemini-prompt-expander) block:

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


## Step 2: Generate and Push the Mellon Config

After adding metadata to your block, generate the default Mellon configuration template and push it to the Hub:

```python
from diffusers.modular_pipelines.mellon_node_utils import MellonPipelineConfig

# Generate the default config template (assume you already loaded your custom blocks into `blocks`)
mellon_config = MellonPipelineConfig.from_custom_block(blocks)
# push the default template to `repo_id`, you will also need to provide a local_dir for it to save it locally
mellon_config.save(
    local_dir="/path/local/folder",
    repo_id= repo_id,
    push_to_hub=True
)
```

This creates a `mellon_pipeline_config.json` file in your repository.

## Step 3: Review and Adjust the Config (Optional)

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

In this example: `prompt` is the only input, there are no model components, and outputs include `out_prompt`, `old_prompt`, and `doc`.

Now let's look at the `params` dict:

**`prompt`** is an input parameter. It has `display: "textarea"` which renders as a text input box, `label: "Prompt"` shown in the UI, and `default: ""` so it starts empty. The `type: "string"` field is important in Mellon because it determines which nodes can connect together - only matching types can be linked with "noodles".

**`out_prompt`** is the expanded prompt output. The `out_` prefix was automatically added because the input and output share the same name (`prompt`), avoiding naming conflicts in the config. It has `display: "output"` which renders as an output socket.

**`old_prompt`** has `type: "custom"` because we didn't specify metadata. This renders as a simple dot in the UI. Since we don't actually want to expose this in the UI, we can remove it.

**`doc`** is the documentation output, automatically added to all custom blocks.

### Making Adjustments

For the Gemini Prompt Expander, we don't need `old_prompt` in the UI. Remove it from both `params` and `output_names`:

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

See the final config at [YiYiXu/gemini-prompt-expander](https://huggingface.co/YiYiXu/gemini-prompt-expander).

## Use in Mellon

1. Start Mellon (see [Mellon installation guide](https://github.com/cubiq/Mellon))

2. In Mellon:
   - Drag a **Dynamic Block Node** from the ModularDiffusers section
   - Enter your `repo_id` (e.g., `YiYiXu/gemini-prompt-expander`)
   - Click **Load Custom Block**
   - The node will transform to show your block's inputs and outputs