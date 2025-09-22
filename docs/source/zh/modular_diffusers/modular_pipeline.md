<!--版权 2025 HuggingFace 团队。保留所有权利。

根据 Apache 许可证 2.0 版（“许可证”）授权；除非符合许可证的规定，否则不得使用此文件。您可以在

http://www.apache.org/licenses/LICENSE-2.0

获取许可证的副本。

除非适用法律要求或书面同意，根据许可证分发的软件是基于“按原样”分发的，没有任何形式的明示或暗示的保证或条件。有关许可证的特定语言，请参阅许可证。
-->

# 模块化管道

[`ModularPipeline`] 将 [`~modular_pipelines.ModularPipelineBlocks`] 转换为可执行的管道，加载模型并执行块中定义的计算步骤。它是运行管道的主要接口，与 [`DiffusionPipeline`] API 非常相似。

主要区别在于在管道中包含了一个预期的 `output` 参数。

<hfoptions id="example">
<hfoption id="text-to-image">

```py
import torch
from diffusers.modular_pipelines import SequentialPipelineBlocks
from diffusers.modular_pipelines.stable_diffusion_xl import TEXT2IMAGE_BLOCKS

blocks = SequentialPipelineBlocks.from_blocks_dict(TEXT2IMAGE_BLOCKS)

modular_repo_id = "YiYiXu/modular-loader-t2i-0704"
pipeline = blocks.init_pipeline(modular_repo_id)

pipeline.load_components(torch_dtype=torch.float16)
pipeline.to("cuda")

image = pipeline(prompt="Astronaut in a jungle, cold color palette, muted colors, detailed, 8k", output="images")[0]
image.save("modular_t2i_out.png")
```

</hfoption>
<hfoption id="image-to-image">

```py
import torch
from diffusers.modular_pipelines import SequentialPipelineBlocks
from diffusers.modular_pipelines.stable_diffusion_xl import IMAGE2IMAGE_BLOCKS

blocks = SequentialPipelineBlocks.from_blocks_dict(IMAGE2IMAGE_BLOCKS)

modular_repo_id = "YiYiXu/modular-loader-t2i-0704"
pipeline = blocks.init_pipeline(modular_repo_id)

pipeline.load_components(torch_dtype=torch.float16)
pipeline.to("cuda")

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/sdxl-text2img.png"
init_image = load_image(url)
prompt = "a dog catching a frisbee in the jungle"
image = pipeline(prompt=prompt, image=init_image, strength=0.8, output="images")[0]
image.save("modular_i2i_out.png")
```

</hfoption>
<hfoption id="inpainting">

```py
import torch
from diffusers.modular_pipelines import SequentialPipelineBlocks
from diffusers.modular_pipelines.stable_diffusion_xl import INPAINT_BLOCKS
from diffusers.utils import load_image

blocks = SequentialPipelineBlocks.from_blocks_dict(INPAINT_BLOCKS)

modular_repo_id = "YiYiXu/modular-loader-t2i-0704"
pipeline = blocks.init_pipeline(modular_repo_id)

pipeline.load_components(torch_dtype=torch.float16)
pipeline.to("cuda")

img_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/sdxl-text2img.png"
mask_url = "h
ttps://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/sdxl-inpaint-mask.png"

init_image = load_image(img_url)
mask_image = load_image(mask_url)

prompt = "A deep sea diver floating"
image = pipeline(prompt=prompt, image=init_image, mask_image=mask_image, strength=0.85, output="images")[0]
image.save("moduar_inpaint_out.png")
```

</hfoption>
</hfoptions>

本指南将向您展示如何创建一个[`ModularPipeline`]并管理其中的组件。

## 添加块

块是[`InsertableDict`]对象，可以在特定位置插入，提供了一种灵活的方式来混合和匹配块。

使用[`~modular_pipelines.modular_pipeline_utils.InsertableDict.insert`]在块类或`sub_blocks`属性上添加一个块。

```py
# BLOCKS是块类的字典，您需要向其中添加类
BLOCKS.insert("block_name", BlockClass, index)
# sub_blocks属性包含实例，向该属性添加一个块实例
t2i_blocks.sub_blocks.insert("block_name", block_instance, index)
```

使用[`~modular_pipelines.modular_pipeline_utils.InsertableDict.pop`]在块类或`sub_blocks`属性上移除一个块。

```py
# 从预设中移除一个块类
BLOCKS.pop("text_encoder")
# 分离出一个块实例
text_encoder_block = t2i_blocks.sub_blocks.pop("text_encoder")
```

通过将现有块设置为新块来交换块。

```py
# 在预设中替换块类
BLOCKS["prepare_latents"] = CustomPrepareLatents
# 使用块实例在sub_blocks属性中替换
t2i_blocks.sub_blocks["prepare_latents"] = CustomPrepareLatents()
```

## 创建管道

有两种方法可以创建一个[`ModularPipeline`]。从[`ModularPipelineBlocks`]组装并创建管道，或使用[`~ModularPipeline.from_pretrained`]加载现有管道。

您还应该初始化一个[`ComponentsManager`]来处理设备放置和内存以及组件管理。

> [!TIP]
> 有关它如何帮助管理不同工作流中的组件的更多详细信息，请参阅[ComponentsManager](./components_manager)文档。

<hfoptions id="create">
<hfoption id="ModularPipelineBlocks">

使用[`~ModularPipelineBlocks.init_pipeline`]方法从组件和配置规范创建一个[`ModularPipeline`]。此方法从`modular_model_index.json`文件加载*规范*，但尚未加载*模型*。

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
<hfoption id="from_pretrained">

[`~ModularPipeline.from_pretrained`]方法创建一个[`ModularPipeline`]从Hub上的模块化仓库加载。

```py
from diffusers import ModularPipeline, ComponentsManager

components = ComponentsManager()
pipeline = ModularPipeline.from_pretrained("YiYiXu/modular-loader-t2i-0704", components_manager=components)
```

添加`trust_remote_code`参数以加载自定义的[`ModularPipeline`]。

```py
from diffusers import ModularPipeline, ComponentsManager

components = ComponentsManager()
modular_repo_id = "YiYiXu/modular-diffdiff-0704"
diffdiff_pipeline = ModularPipeline.from_pretrained(modular_repo_id, trust_remote_code=True, components_manager=components)
```

</hfoption>
</hfoptions>

## 加载组件

一个[`ModularPipeline`]不会自动实例化组件。它只加载配置和组件规范。您可以使用[`~ModularPipeline.load_components`]加载所有组件，或仅使用[`~ModularPipeline.load_components`]加载特定组件。

<hfoptions id="load">
<hfoption id="load_components">

```py
import torch

t2i_pipeline.load_components(torch_dtype=torch.float16)
t2i_pipeline.to("cuda")
```

</hfoption>
<hfoption id="load_components">

下面的例子仅加载UNet和VAE。

```py
import torch

t2i_pipeline.load_components(names=["unet", "vae"], torch_dtype=torch.float16)
```

</hfoption>
</hfoptions>

打印管道以检查加载的预训练组件。

```py
t2i_pipeline
```

这应该与管道初始化自的模块化仓库中的`modular_model_index.json`文件匹配。如果管道不需要某个组件，即使它在模块化仓库中存在，也不会被包含。

要修改组件加载的来源，编辑仓库中的`modular_model_index.json`文件，并将其更改为您希望的加载路径。下面的例子从不同的仓库加载UNet。

```json
# 原始
"unet": [
  null, null,
  {
    "repo": "stabilityai/stable-diffusion-xl-base-1.0",
    "subfolder": "unet",
    "variant": "fp16"
  }
]

# 修改后
"unet": [
  null, null,
  {
    "repo": "RunDiffusion/Juggernaut-XL-v9",
    "subfolder": "unet",
    "variant": "fp16"
  }
]
```

### 组件加载状态

下面的管道属性提供了关于哪些组件被加载的更多信息。

使用`component_names`返回所有预期的组件。

```py
t2i_pipeline.component_names
['text_encoder', 'text_encoder_2', 'tokenizer', 'tokenizer_2', 'guider', 'scheduler', 'unet', 'vae', 'image_processor']
```

使用`null_component_names`返回尚未加载的组件。使用[`~ModularPipeline.from_pretrained`]加载这些组件。

```py
t2i_pipeline.null_component_names
['text_encoder', 'text_encoder_2', 'tokenizer', 'tokenizer_2', 'scheduler']
```

使用`pretrained_component_names`返回将从预训练模型加载的组件。

```py
t2i_pipeline.pretrained_component_names
['text_encoder', 'text_encoder_2', 'tokenizer', 'tokenizer_2', 'scheduler', 'unet', 'vae']
```

使用 `config_component_names` 返回那些使用默认配置创建的组件（不是从模块化仓库加载的）。来自配置的组件不包括在内，因为它们已经在管道创建期间初始化。这就是为什么它们没有列在 `null_component_names` 中。

```py
t2i_pipeline.config_component_names
['guider', 'image_processor']
```

## 更新组件

根据组件是*预训练组件*还是*配置组件*，组件可能会被更新。

> [!WARNING]
> 在更新组件时，组件可能会从预训练变为配置。组件类型最初是在块的 `expected_components` 字段中定义的。

预训练组件通过 [`ComponentSpec`] 更新，而配置组件则通过直接传递对象或使用 [`ComponentSpec`] 更新。

[`ComponentSpec`] 对于预训练组件显示 `default_creation_method="from_pretrained"`，对于配置组件显示 `default_creation_method="from_config`。

要更新预训练组件，创建一个 [`ComponentSpec`]，指定组件的名称和从哪里加载它。使用 [`~ComponentSpec.load`] 方法来加载组件。

```py
from diffusers import ComponentSpec, UNet2DConditionModel

unet_spec = ComponentSpec(name="unet",type_hint=UNet2DConditionModel, repo="stabilityai/stable-diffusion-xl-base-1.0", subfolder="unet", variant="fp16")
unet = unet_spec.load(torch_dtype=torch.float16)
```

[`~ModularPipeline.update_components`] 方法用一个新的组件替换原来的组件。

```py
t2i_pipeline.update_components(unet=unet2)
```

当组件被更新时，加载规范也会在管道配置中更新。

### 组件提取和修改

当你使用 [`~ComponentSpec.load`] 时，新组件保持其加载规范。这使得提取规范并重新创建组件成为可能。

```py
spec = ComponentSpec.from_component("unet", unet2)
spec
ComponentSpec(name='unet', type_hint=<class 'diffusers.models.unets.unet_2d_condition.UNet2DConditionModel'>, description=None, config=None, repo='stabilityai/stable-diffusion-xl-base-1.0', subfolder='unet', variant='fp16', revision=None, default_creation_method='from_pretrained')
unet2_recreated = spec.load(torch_dtype=torch.float16)
```

[`~ModularPipeline.get_component_spec`] 方法获取当前组件规范的副本以进行修改或更新。

```py
unet_spec = t2i_pipeline.get_component_spec("unet")
unet_spec
ComponentSpec(
    name='unet',
    type_hint=<class 'diffusers.models.unets.unet_2d_condition.UNet2DConditionModel'>,
    repo='RunDiffusion/Juggernaut-XL-v9',
    subfolder='unet',
    variant='fp16',
    default_creation_method='from_pretrained'
)

# 修改以从不同的仓库加载
unet_spec.repo = "stabilityai/stable-diffusion-xl-base-1.0"

# 使用修改后的规范加载组件
unet = unet_spec.load(torch_dtype=torch.float16)
```

## 模块化仓库
一个仓库
如果管道块使用*预训练组件*，则需要y。该存储库提供了加载规范和元数据。

[`ModularPipeline`]特别需要*模块化存储库*（参见[示例存储库](https://huggingface.co/YiYiXu/modular-diffdiff)），这比典型的存储库更灵活。它包含一个`modular_model_index.json`文件，包含以下3个元素。

- `library`和`class`显示组件是从哪个库加载的及其类。如果是`null`，则表示组件尚未加载。
- `loading_specs_dict`包含加载组件所需的信息，例如从中加载的存储库和子文件夹。

与标准存储库不同，模块化存储库可以根据`loading_specs_dict`从不同的存储库获取组件。组件不需要存在于同一个存储库中。

模块化存储库可能包含用于加载[`ModularPipeline`]的自定义代码。这允许您使用不是Diffusers原生的专用块。

```
modular-diffdiff-0704/
├── block.py                    # 自定义管道块实现
├── config.json                 # 管道配置和auto_map
└── modular_model_index.json    # 组件加载规范
```

[config.json](https://huggingface.co/YiYiXu/modular-diffdiff-0704/blob/main/config.json)文件包含一个`auto_map`键，指向`block.py`中定义自定义块的位置。

```json
{
  "_class_name": "DiffDiffBlocks",
  "auto_map": {
    "ModularPipelineBlocks": "block.DiffDiffBlocks"
  }
}
```
