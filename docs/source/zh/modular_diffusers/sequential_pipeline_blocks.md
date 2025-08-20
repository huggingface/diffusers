<!--版权 2025 The HuggingFace Team。保留所有权利。

根据Apache许可证2.0版（"许可证"）授权；除非符合许可证，否则不得使用此文件。您可以在

http://www.apache.org/licenses/LICENSE-2.0

获取许可证的副本。

除非适用法律要求或书面同意，根据许可证分发的软件是基于"按原样"基础分发的，没有任何形式的明示或暗示的保证或条件。有关许可证下特定语言的管理权限和限制，请参阅许可证。
-->

# 顺序管道块

[`~modular_pipelines.SequentialPipelineBlocks`] 是一种多块类型，它将其他 [`~modular_pipelines.ModularPipelineBlocks`] 按顺序组合在一起。数据通过 `intermediate_inputs` 和 `intermediate_outputs` 线性地从一个块流向下一个块。[`~modular_pipelines.SequentialPipelineBlocks`] 中的每个块通常代表管道中的一个步骤，通过组合它们，您逐步构建一个管道。

本指南向您展示如何将两个块连接成一个 [`~modular_pipelines.SequentialPipelineBlocks`]。

创建两个 [`~modular_pipelines.ModularPipelineBlocks`]。第一个块 `InputBlock` 输出一个 `batch_size` 值，第二个块 `ImageEncoderBlock` 使用 `batch_size` 作为 `intermediate_inputs`。

<hfoptions id="sequential">
<hfoption id="InputBlock">

```py
from diffusers.modular_pipelines import ModularPipelineBlocks, InputParam, OutputParam

class InputBlock(ModularPipelineBlocks):

    @property
    def inputs(self):
        return [
            InputParam(name="prompt", type_hint=list, description="list of text prompts"),
            InputParam(name="num_images_per_prompt", type_hint=int, description="number of images per prompt"),
        ]

    @property
    def intermediate_outputs(self):
        return [
            OutputParam(name="batch_size", description="calculated batch size"),
        ]

    @property
    def description(self):
        return "A block that determines batch_size based on the number of prompts and num_images_per_prompt argument."

    def __call__(self, components, state):
        block_state = self.get_block_state(state)
        batch_size = len(block_state.prompt)
        block_state.batch_size = batch_size * block_state.num_images_per_prompt
        self.set_block_state(state, block_state)
        return components, state
```

</hfoption>
<hfoption id="ImageEncoderBlock">

```py
import torch
from diffusers.modular_pipelines import ModularPipelineBlocks, InputParam, OutputParam

class ImageEncoderBlock(ModularPipelineBlocks):

    @property
    def inputs(self):
        return [
            InputParam(name="image", type_hint="PIL.Image", description="raw input image to process"),
            InputParam(name="batch_size", type_hint=int),
        ]

    @property
    def intermediate_outputs(self):
        return [
            OutputParam(name="image_latents", description="latents representing the image"
        ]

    @property
    def description(self):
        return "Encode raw image into its latent presentation"

    def __call__(self, components, state):
        block_state = self.get_block_state(state)
        # 模拟处理图像
        # 这将改变所有块的图像状态，从PIL图像变为张量
        block_state.image = torch.randn(1, 3, 512, 512)
        block_state.batch_size = block_state.batch_size * 2
        block_state.image_latents = torch.randn(1, 4, 64, 64)
        self.set_block_state(state, block_state)
        return components, state
```

</hfoption>
</hfoptions>

通过定义一个[`InsertableDict`]来连接两个块，将块名称映射到块实例。块按照它们在`blocks_dict`中注册的顺序执行。

使用[`~modular_pipelines.SequentialPipelineBlocks.from_blocks_dict`]来创建一个[`~modular_pipelines.SequentialPipelineBlocks`]。

```py
from diffusers.modular_pipelines import SequentialPipelineBlocks, InsertableDict

blocks_dict = InsertableDict()
blocks_dict["input"] = input_block
blocks_dict["image_encoder"] = image_encoder_block

blocks = SequentialPipelineBlocks.from_blocks_dict(blocks_dict)
```

通过调用`blocks`来检查[`~modular_pipelines.SequentialPipelineBlocks`]中的子块，要获取更多关于输入和输出的详细信息，可以访问`docs`属性。

```py
print(blocks)
print(blocks.doc)
```
