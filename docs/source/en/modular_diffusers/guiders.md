<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Guiders

[Classifier-free guidance](https://huggingface.co/papers/2207.12598) steers model generation that better match a prompt and is commonly used to improve generation quality, control, and adherence to prompts. There are different types of guidance methods, and in Diffusers, they are known as *guiders*. Like blocks, it is easy to switch and use different guiders for different use cases without rewriting the pipeline.

This guide will show you how to switch guiders, adjust guider parameters, and load and share them to the Hub.

## Switching guiders

[`ClassifierFreeGuidance`] is the default guider and created when a pipeline is initialized with [`~ModularPipelineBlocks.init_pipeline`]. It is created by `from_config` which means it doesn't require loading specifications from a modular repository. A guider won't be listed in `modular_model_index.json`.

Use [`~ModularPipeline.get_component_spec`] to inspect a guider.

```py
t2i_pipeline.get_component_spec("guider")
ComponentSpec(name='guider', type_hint=<class 'diffusers.guiders.classifier_free_guidance.ClassifierFreeGuidance'>, description=None, config=FrozenDict([('guidance_scale', 7.5), ('guidance_rescale', 0.0), ('use_original_formulation', False), ('start', 0.0), ('stop', 1.0), ('_use_default_values', ['start', 'guidance_rescale', 'stop', 'use_original_formulation'])]), repo=None, subfolder=None, variant=None, revision=None, default_creation_method='from_config')
```

Switch to a different guider by passing the new guider to [`~ModularPipeline.update_components`].

> [!TIP]
> Changing guiders will return text letting you know you're changing the guider type.
> ```bash
> ModularPipeline.update_components: adding guider with new type: PerturbedAttentionGuidance, previous type: ClassifierFreeGuidance
> ```

```py
from diffusers import LayerSkipConfig, PerturbedAttentionGuidance

config = LayerSkipConfig(indices=[2, 9], fqn="mid_block.attentions.0.transformer_blocks", skip_attention=False, skip_attention_scores=True, skip_ff=False)
guider = PerturbedAttentionGuidance(
    guidance_scale=5.0, perturbed_guidance_scale=2.5, perturbed_guidance_config=config
)
t2i_pipeline.update_components(guider=guider)
```

Use [`~ModularPipeline.get_component_spec`] again to verify the guider type is different.

```py
t2i_pipeline.get_component_spec("guider")
ComponentSpec(name='guider', type_hint=<class 'diffusers.guiders.perturbed_attention_guidance.PerturbedAttentionGuidance'>, description=None, config=FrozenDict([('guidance_scale', 5.0), ('perturbed_guidance_scale', 2.5), ('perturbed_guidance_start', 0.01), ('perturbed_guidance_stop', 0.2), ('perturbed_guidance_layers', None), ('perturbed_guidance_config', LayerSkipConfig(indices=[2, 9], fqn='mid_block.attentions.0.transformer_blocks', skip_attention=False, skip_attention_scores=True, skip_ff=False, dropout=1.0)), ('guidance_rescale', 0.0), ('use_original_formulation', False), ('start', 0.0), ('stop', 1.0), ('_use_default_values', ['perturbed_guidance_start', 'use_original_formulation', 'perturbed_guidance_layers', 'stop', 'start', 'guidance_rescale', 'perturbed_guidance_stop']), ('_class_name', 'PerturbedAttentionGuidance'), ('_diffusers_version', '0.35.0.dev0')]), repo=None, subfolder=None, variant=None, revision=None, default_creation_method='from_config')
```

## Loading custom guiders

Guiders that are already saved on the Hub with a `modular_model_index.json` file are considered a `from_pretrained` component now instead of a `from_config` component.

```json
{
  "guider": [
    null,
    null,
    {
      "repo": "YiYiXu/modular-loader-t2i-guider",
      "revision": null,
      "subfolder": "pag_guider",
      "type_hint": [
        "diffusers",
        "PerturbedAttentionGuidance"
      ],
      "variant": null
    }
  ]
}
```

The guider is only created after calling [`~ModularPipeline.load_components`] based on the loading specification in `modular_model_index.json`.

```py
t2i_pipeline = t2i_blocks.init_pipeline("YiYiXu/modular-doc-guider")
# not created during init
assert t2i_pipeline.guider is None
t2i_pipeline.load_components()
# loaded as PAG guider
t2i_pipeline.guider
```


## Changing guider parameters

The guider parameters can be adjusted with either the [`~ComponentSpec.create`] method or with [`~ModularPipeline.update_components`]. The example below changes the `guidance_scale` value.

<hfoptions id="switch">
<hfoption id="create">

```py
guider_spec = t2i_pipeline.get_component_spec("guider")
guider = guider_spec.create(guidance_scale=10)
t2i_pipeline.update_components(guider=guider)
```

</hfoption>
<hfoption id="update_components">

```py
guider_spec = t2i_pipeline.get_component_spec("guider")
guider_spec.config["guidance_scale"] = 10
t2i_pipeline.update_components(guider=guider_spec)
```

</hfoption>
</hfoptions>

## Uploading custom guiders

Call the [`~utils.PushToHubMixin.push_to_hub`] method on a custom guider to share it to the Hub.

```py
guider.push_to_hub("YiYiXu/modular-loader-t2i-guider", subfolder="pag_guider")
```

To make this guider available to the pipeline, either modify the `modular_model_index.json` file or use the [`~ModularPipeline.update_components`] method.

<hfoptions id="upload">
<hfoption id="modular_model_index.json">

Edit the `modular_model_index.json` file and add a loading specification for the guider by pointing to a folder containing the guider config.

```json
{
  "guider": [
    "diffusers",
    "PerturbedAttentionGuidance",
    {
      "repo": "YiYiXu/modular-loader-t2i-guider",
      "revision": null,
      "subfolder": "pag_guider",
      "type_hint": [
        "diffusers",
        "PerturbedAttentionGuidance"
      ],
      "variant": null
    }
  ],
```

</hfoption>
<hfoption id="update_components">

Change the [`~ComponentSpec.default_creation_method`] to `from_pretrained` and use [`~ModularPipeline.update_components`] to update the guider and component specifications as well as the pipeline config.

> [!TIP]
> Changing the creation method will return text letting you know you're changing the creation type to `from_pretrained`.
> ```bash
> ModularPipeline.update_components: changing the default_creation_method of guider from from_config to from_pretrained.
> ```

```py
guider_spec = t2i_pipeline.get_component_spec("guider")
guider_spec.default_creation_method="from_pretrained"
guider_spec.repo="YiYiXu/modular-loader-t2i-guider"
guider_spec.subfolder="pag_guider"
pag_guider = guider_spec.load()
t2i_pipeline.update_components(guider=pag_guider)
```

To make it the default guider for a pipeline, call [`~utils.PushToHubMixin.push_to_hub`]. This is an optional step and not necessary if you are only experimenting locally.

```py
t2i_pipeline.push_to_hub("YiYiXu/modular-doc-guider")
```

</hfoption>
</hfoptions>
