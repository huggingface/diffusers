<!--版权所有 2025 HuggingFace 团队。保留所有权利。

根据 Apache 许可证 2.0 版（"许可证"）授权；除非遵守许可证，否则不得使用此文件。您可以在以下网址获取许可证副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据许可证分发的软件按"原样"分发，无任何明示或暗示的担保或条件。请参阅许可证以了解特定语言管理权限和限制。
-->

# 组件管理器

[`ComponentsManager`] 是 Modular Diffusers 的模型注册和管理系统。它添加和跟踪模型，存储有用的元数据（模型大小、设备放置、适配器），防止重复模型实例，并支持卸载。

本指南将展示如何使用 [`ComponentsManager`] 来管理组件和设备内存。

## 添加组件

[`ComponentsManager`] 应与 [`ModularPipeline`] 一起创建，在 [`~ModularPipeline.from_pretrained`] 或 [`~ModularPipelineBlocks.init_pipeline`] 中。

> [!TIP]
> `collection` 参数是可选的，但可以更轻松地组织和管理组件。

<hfoptions id="create">
<hfoption id="from_pretrained">

```py
from diffusers import ModularPipeline, ComponentsManager

comp = ComponentsManager()
pipe = ModularPipeline.from_pretrained("YiYiXu/modular-demo-auto", components_manager=comp, collection="test1")
```

</hfoption>
<hfoption id="init_pipeline">

```py
from diffusers import ComponentsManager
from diffusers.modular_pipelines import SequentialPipelineBlocks
from diffusers.modular_pipelines.stable_diffusion_xl import TEXT2IMAGE_BLOCKS

t2i_blocks = SequentialPipelineBlocks.from_blocks_dict(TEXT2IMAGE_BLOCKS)

modular_repo_id = "YiYiXu/modular-loader-t2i-0704"
components = ComponentsManager()
t2i_pipeline = t2i_blocks.init_pipeline(modular_repo_id, components_manager=components)
```

</hfoption>
</hfoptions>

组件仅在调用 [`~ModularPipeline.load_components`] 或 [`~ModularPipeline.load_components`] 时加载和注册。以下示例使用 [`~ModularPipeline.load_components`] 创建第二个管道，重用第一个管道的所有组件，并将其分配到不同的集合。

```py
pipe.load_components()
pipe2 = ModularPipeline.from_pretrained("YiYiXu/modular-demo-auto", components_manager=comp, collection="test2")
```

使用 [`~ModularPipeline.null_component_names`] 属性来识别需要加载的任何组件，使用 [`~ComponentsManager.get_components_by_names`] 检索它们，然后调用 [`~ModularPipeline.update_components`] 来添加缺失的组件。

```py
pipe2.null_component_names 
['text_encoder', 'text_encoder_2', 'tokenizer', 'tokenizer_2', 'image_encoder', 'unet', 'vae', 'scheduler', 'controlnet']

comp_dict = comp.get_components_by_names(names=pipe2.null_component_names)
pipe2.update_components(**comp_dict)
```

要添加单个组件，请使用 [`~ComponentsManager.add`] 方法。这会使用唯一 id 注册一个组件。

```py
from diffusers import AutoModel

text_encoder = AutoModel.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="text_encoder")
component_id = comp.add("text_encoder", text_encoder)
comp
```

使用 [`~ComponentsManager.remove`] 通过其 id 移除一个组件。

```py
comp.remove("text_encoder_139917733042864")
```

## 检索组件

[`ComponentsManager`] 提供了几种方法来检索已注册的组件。

### get_one

[`~ComponentsManager.get_one`] 方法返回单个组件，并支持对 `name` 参数进行模式匹配。如果多个组件匹配，[`~ComponentsManager.get_one`] 会返回错误。

| 模式       | 示例                             | 描述                                   |
|-------------|----------------------------------|-------------------------------------------|
| exact       | `comp.get_one(name="unet")`      | 精确名称匹配                          |
| wildcard    | `comp.get_one(name="unet*")`     | 名称以 "unet" 开头                |
| exclusion   | `comp.get_one(name="!unet")`     | 排除名为 "unet" 的组件           |
| or          | `comp.get_one(name="unet&#124;vae")`  | 名称为 "unet" 或 "vae"                   |

[`~ComponentsManager.get_one`] 还通过 `collection` 参数或 `load_id` 参数过滤组件。

```py
comp.get_one(name="unet", collection="sdxl")
```

### get_components_by_names

[`~ComponentsManager.get_components_by_names`] 方法接受一个名称列表，并返回一个将名称映射到组件的字典。这在 [`ModularPipeline`] 中特别有用，因为它们提供了所需组件名称的列表，并且返回的字典可以直接传递给 [`~ModularPipeline.update_components`]。

```py
component_dict = comp.get_components_by_names(names=["text_encoder", "unet", "vae"])
{"text_encoder": component1, "unet": component2, "vae": component3}
```

## 重复检测

建议使用 [`ComponentSpec`] 加载模型组件，以分配具有唯一 id 的组件，该 id 编码了它们的加载参数。这允许 [`ComponentsManager`] 自动检测并防止重复的模型实例，即使不同的对象代表相同的底层检查点。

```py
from diffusers import ComponentSpec, ComponentsManager
from transformers import CLIPTextModel

comp = ComponentsManager()

# 为第一个文本编码器创建 ComponentSpec
spec = ComponentSpec(name="text_encoder", repo="stabilityai/stable-diffusion-xl-base-1.0", subfolder="text_encoder", type_hint=AutoModel)
# 为重复的文本编码器创建 ComponentSpec（它是相同的检查点，来自相同的仓库/子文件夹）
spec_duplicated = ComponentSpec(name="text_encoder_duplicated", repo="stabilityai/stable-diffusion-xl-base-1.0", subfolder="text_encoder", ty
pe_hint=CLIPTextModel)

# 加载并添加两个组件 - 管理器会检测到它们是同一个模型
comp.add("text_encoder", spec.load())
comp.add("text_encoder_duplicated", spec_duplicated.load())
```

这会返回一个警告，附带移除重复项的说明。

```py
ComponentsManager: adding component 'text_encoder_duplicated_139917580682672', but it has duplicate load_id 'stabilityai/stable-diffusion-xl-base-1.0|text_encoder|null|null' with existing components: text_encoder_139918506246832. To remove a duplicate, call `components_manager.remove('<component_id>')`.
'text_encoder_duplicated_139917580682672'
```

您也可以不使用 [`ComponentSpec`] 添加组件，并且在大多数情况下，即使您以不同名称添加相同组件，重复检测仍然有效。

然而，当您将相同组件加载到不同对象时，[`ComponentManager`] 无法检测重复项。在这种情况下，您应该使用 [`ComponentSpec`] 加载模型。

```py
text_encoder_2 = AutoModel.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="text_encoder")
comp.add("text_encoder", text_encoder_2)
'text_encoder_139917732983664'
```

## 集合

集合是为组件分配的标签，用于更好的组织和管理。使用 [`~ComponentsManager.add`] 中的 `collection` 参数将组件添加到集合中。

每个集合中只允许每个名称有一个组件。添加第二个同名组件会自动移除第一个组件。

```py
from diffusers import ComponentSpec, ComponentsManager

comp = ComponentsManager()
# 为第一个 UNet 创建 ComponentSpec
spec = ComponentSpec(name="unet", repo="stabilityai/stable-diffusion-xl-base-1.0", subfolder="unet", type_hint=AutoModel)
# 为另一个 UNet 创建 ComponentSpec
spec2 = ComponentSpec(name="unet", repo="RunDiffusion/Juggernaut-XL-v9", subfolder="unet", type_hint=AutoModel, variant="fp16")

# 将两个 UNet 添加到同一个集合 - 第二个将替换第一个
comp.add("unet", spec.load(), collection="sdxl")
comp.add("unet", spec2.load(), collection="sdxl")
```

这使得在基于节点的系统中工作变得方便，因为您可以：

- 使用 `collection` 标签标记所有从一个节点加载的模型。
- 当新检查点以相同名称加载时自动替换模型。
- 当节点被移除时批量删除集合中的所有模型。

## 卸载

[`~ComponentsManager.enable_auto_cpu_offload`] 方法是一种全局卸载策略，适用于所有模型，无论哪个管道在使用它们。一旦启用，您无需担心设备放置，如果您添加或移除组件。

```py
comp.enable_auto_cpu_offload(device="cuda")
```

所有模型开始时都在 CPU 上，[`ComponentsManager`] 在需要它们之前将它们移动到适当的设备，并在 GPU 内存不足时将其他模型移回 CPU。

您可以设置自己的规则来决定哪些模型要卸载。
