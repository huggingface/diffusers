<!--版权所有 2025 The HuggingFace Team。保留所有权利。

根据Apache许可证2.0版（"许可证"）授权；除非符合许可证，否则不得使用此文件。您可以在

http://www.apache.org/licenses/LICENSE-2.0

获取许可证的副本。

除非适用法律要求或书面同意，根据许可证分发的软件按"原样"分发，无任何明示或暗示的担保或条件。有关许可证的特定语言管理权限和限制，请参阅许可证。
-->

# AutoPipelineBlocks

[`~modular_pipelines.AutoPipelineBlocks`] 是一种包含支持不同工作流程的块的多块类型。它根据运行时提供的输入自动选择要运行的子块。这通常用于将多个工作流程（文本到图像、图像到图像、修复）打包到一个管道中以便利。

本指南展示如何创建 [`~modular_pipelines.AutoPipelineBlocks`]。

创建三个 [`~modular_pipelines.ModularPipelineBlocks`] 用于文本到图像、图像到图像和修复。这些代表了管道中可用的不同工作流程。

<hfoptions id="auto">
<hfoption id="text-to-image">

```py
import torch
from diffusers.modular_pipelines import ModularPipelineBlocks, InputParam, OutputParam

class TextToImageBlock(ModularPipelineBlocks):
    model_name = "text2img"

    @property
    def inputs(self):
        return [InputParam(name="prompt")]

    @property
    def intermediate_outputs(self):
        return []

    @property
    def description(self):
        return "我是一个文本到图像的工作流程！"

    def __call__(self, components, state):
        block_state = self.get_block_state(state)
        print("运行文本到图像工作流程")
        # 在这里添加你的文本到图像逻辑
        # 例如：根据提示生成图像
        self.set_block_state(state, block_state)
        return components, state
```


</hfoption>
<hfoption id="image-to-image">

```py
class ImageToImageBlock(ModularPipelineBlocks):
    model_name = "img2img"

    @property
    def inputs(self):
        return [InputParam(name="prompt"), InputParam(name="image")]

    @property
    def intermediate_outputs(self):
        return []

    @property
    def description(self):
        return "我是一个图像到图像的工作流程！"

    def __call__(self, components, state):
        block_state = self.get_block_state(state)
        print("运行图像到图像工作流程")
        # 在这里添加你的图像到图像逻辑
        # 例如：根据提示转换输入图像
        self.set_block_state(state, block_state)
        return components, state
```


</hfoption>
<hfoption id="inpaint">

```py
class InpaintBlock(ModularPipelineBlocks):
    model_name = "inpaint"

    @property
    def inputs(self):
        return [InputParam(name="prompt"), InputParam(name="image"), InputParam(name="mask")]

    @property

    def intermediate_outputs(self):
        return []

    @property
    def description(self):
        return "我是一个修复工作流！"

    def __call__(self, components, state):
        block_state = self.get_block_state(state)
        print("运行修复工作流")
        # 在这里添加你的修复逻辑
        # 例如：根据提示填充被遮罩的区域
        self.set_block_state(state, block_state)
        return components, state
```

</hfoption>
</hfoptions>

创建一个包含子块类及其对应块名称列表的[`~modular_pipelines.AutoPipelineBlocks`]类。

你还需要包括`block_trigger_inputs`，一个触发相应块的输入名称列表。如果在运行时提供了触发输入，则选择该块运行。使用`None`来指定如果未检测到触发输入时运行的默认块。

最后，重要的是包括一个`description`，清楚地解释哪些输入触发哪些工作流。这有助于用户理解如何运行特定的工作流。

```py
from diffusers.modular_pipelines import AutoPipelineBlocks

class AutoImageBlocks(AutoPipelineBlocks):
    # 选择子块类的列表
    block_classes = [block_inpaint_cls, block_i2i_cls, block_t2i_cls]
    # 每个块的名称，顺序相同
    block_names = ["inpaint", "img2img", "text2img"]
    # 决定运行哪个块的触发输入
    # - "mask" 触发修复工作流
    # - "image" 触发img2img工作流（但仅在未提供mask时）
    # - 如果以上都没有，运行text2img工作流（默认）
    block_trigger_inputs = ["mask", "image", None]
    # 对于AutoPipelineBlocks来说，描述极其重要

    def description(self):
        return (
            "Pipeline generates images given different types of conditions!\n"
            + "This is an auto pipeline block that works for text2img, img2img and inpainting tasks.\n"
            + " - inpaint workflow is run when `mask` is provided.\n"
            + " - img2img workflow is run when `image` is provided (but only when `mask` is not provided).\n"
            + " - text2img workflow is run when neither `image` nor `mask` is provided.\n"
        )
```

包含`description`以避免任何关于如何运行块和需要什么输入的混淆**非常**重要。虽然[`~modular_pipelines.AutoPipelineBlocks`]很方便，但如果它没有正确解释，其条件逻辑可能难以理解。

创建`AutoImageBlocks`的一个实例。

```py
auto_blocks = AutoImageBlocks()
```

对于更复杂的组合，例如在更大的管道中作为子块使用的嵌套[`~modular_pipelines.AutoPipelineBlocks`]块，使用[`~modular_pipelines.SequentialPipelineBlocks.get_execution_blocks`]方法根据你的输入提取实际运行的块。

```py
auto_blocks.get_execution_blocks("mask")
```
