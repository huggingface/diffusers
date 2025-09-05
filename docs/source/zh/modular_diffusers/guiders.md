<!--版权所有 2025 The HuggingFace Team。保留所有权利。

根据 Apache 许可证 2.0 版（"许可证"）授权；除非遵守许可证，否则不得使用此文件。
您可以在以下网址获取许可证副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据许可证分发的软件按"原样"分发，不附带任何明示或暗示的担保或条件。请参阅许可证了解具体的语言管理权限和限制。
-->

# 引导器

[Classifier-free guidance](https://huggingface.co/papers/2207.12598) 引导模型生成更好地匹配提示，通常用于提高生成质量、控制和提示的遵循度。有不同类型的引导方法，在 Diffusers 中，它们被称为*引导器*。与块类似，可以轻松切换和使用不同的引导器以适应不同的用例，而无需重写管道。

本指南将向您展示如何切换引导器、调整引导器参数，以及将它们加载并共享到 Hub。

## 切换引导器

[`ClassifierFreeGuidance`] 是默认引导器，在使用 [`~ModularPipelineBlocks.init_pipeline`] 初始化管道时创建。它通过 `from_config` 创建，这意味着它不需要从模块化存储库加载规范。引导器不会列在 `modular_model_index.json` 中。

使用 [`~ModularPipeline.get_component_spec`] 来检查引导器。

```py
t2i_pipeline.get_component_spec("guider")
ComponentSpec(name='guider', type_hint=<class 'diffusers.guiders.classifier_free_guidance.ClassifierFreeGuidance'>, description=None, config=FrozenDict([('guidance_scale', 7.5), ('guidance_rescale', 0.0), ('use_original_formulation', False), ('start', 0.0), ('stop', 1.0), ('_use_default_values', ['start', 'guidance_rescale', 'stop', 'use_original_formulation'])]), repo=None, subfolder=None, variant=None, revision=None, default_creation_method='from_config')
```

通过将新引导器传递给 [`~ModularPipeline.update_components`] 来切换到不同的引导器。

> [!TIP]
> 更改引导器将返回文本，让您知道您正在更改引导器类型。
> ```bash
> ModularPipeline.update_components: 添加具有新类型的引导器: PerturbedAttentionGuidance, 先前类型: ClassifierFreeGuidance
> ```

```py
from diffusers import LayerSkipConfig, PerturbedAttentionGuidance

config = LayerSkipConfig(indices=[2, 9], fqn="mid_block.attentions.0.transformer_blocks", skip_attention=False, skip_attention_scores=True, skip_ff=False)
guider = PerturbedAttentionGuidance(
    guidance_scale=5.0, perturbed_guidance_scale=2.5, perturbed_guidance_config=config
)
t2i_pipeline.update_components(guider=guider)
```

再次使用 [`~ModularPipeline.get_component_spec`] 来验证引导器类型是否不同。

```py
t2i_pipeline.get_component_spec("guider")
ComponentSpec(name='guider', type_hint=<class 'diffusers.guiders.perturbed_attention_guidance.PerturbedAttentionGuidance'>, description=None, config=FrozenDict([('guidance_scale', 5.0), ('perturbed_guidance_scale', 2.5), ('perturbed_guidance_start', 0.01), ('perturbed_guidance_stop', 0.2), ('perturbed_guidance_layers', None), ('perturbed_guidance_config', LayerSkipConfig(indices=[2, 9], fqn='mid_block.attentions.0.transformer_blocks', skip_attention=False, skip_attention_scores=True, skip_ff=False, dropout=1.0)), ('guidance_rescale', 0.0), ('use_original_formulation', False), ('start', 0.0), ('stop', 1.0), ('_use_default_values', ['perturbed_guidance_start', 'use_original_formulation', 'perturbed_guidance_layers', 'stop', 'start', 'guidance_rescale', 'perturbed_guidance_stop']), ('_class_name', 'PerturbedAttentionGuidance'), ('_diffusers_version', '0.35.0.dev0')]), repo=None, subfolder=None, variant=None, revision=None, default_creation_method='from_config')
```

## 加载自定义引导器

已经在 Hub 上保存并带有 `modular_model_index.json` 文件的引导器现在被视为 `from_pretrained` 组件，而不是 `from_config` 组件。

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

引导器只有在调用 [`~ModularPipeline.load_components`] 之后才会创建，基于 `modular_model_index.json` 中的加载规范。

```py
t2i_pipeline = t2i_blocks.init_pipeline("YiYiXu/modular-doc-guider")
# 在初始化时未创建
assert t2i_pipeline.guider is None
t2i_pipeline.load_components()
# 加载为 PAG 引导器
t2i_pipeline.guider
```

## 更改引导器参数

引导器参数可以通过 [`~ComponentSpec.create`] 方法或 [`~ModularPipeline.update_components`] 方法进行调整。下面的示例更改了 `guidance_scale` 值。

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

## 上传自定义引导器

在自定义引导器上调用 [`~utils.PushToHubMixin.push_to_hub`] 方法，将其分享到 Hub。

```py
guider.push_to_hub("YiYiXu/modular-loader-t2i-guider", subfolder="pag_guider")
```

要使此引导器可用于管道，可以修改 `modular_model_index.json` 文件或使用 [`~ModularPipeline.update_components`] 方法。

<hfoptions id="upload">
<hfoption id="modular_model_index.json">

编辑 `modular_model_index.json` 文件，并添加引导器的加载规范，指向包含引导器配置的文件夹
例如。

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

将 [`~ComponentSpec.default_creation_method`] 更改为 `from_pretrained` 并使用 [`~ModularPipeline.update_components`] 来更新引导器和组件规范以及管道配置。

> [!TIP]
> 更改创建方法将返回文本，告知您正在将创建类型更改为 `from_pretrained`。
> ```bash
> ModularPipeline.update_components: 将引导器的 default_creation_method 从 from_config 更改为 from_pretrained。
> ```

```py
guider_spec = t2i_pipeline.get_component_spec("guider")
guider_spec.default_creation_method="from_pretrained"
guider_spec.repo="YiYiXu/modular-loader-t2i-guider"
guider_spec.subfolder="pag_guider"
pag_guider = guider_spec.load()
t2i_pipeline.update_components(guider=pag_guider)
```

要使其成为管道的默认引导器，请调用 [`~utils.PushToHubMixin.push_to_hub`]。这是一个可选步骤，如果您仅在本地进行实验，则不需要。

```py
t2i_pipeline.push_to_hub("YiYiXu/modular-doc-guider")
```

</hfoption>
</hfoptions>
