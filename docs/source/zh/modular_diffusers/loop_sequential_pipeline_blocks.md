<!--版权 2025 The HuggingFace Team。保留所有权利。

根据 Apache 许可证 2.0 版（"许可证"）授权；除非符合许可证，否则不得使用此文件。您可以在

http://www.apache.org/licenses/LICENSE-2.0

获取许可证的副本。

除非适用法律要求或书面同意，根据许可证分发的软件按"原样"分发，无任何明示或暗示的担保或条件。请参阅许可证了解
特定语言下的权限和限制。
-->

# LoopSequentialPipelineBlocks

[`~modular_pipelines.LoopSequentialPipelineBlocks`] 是一种多块类型，它将其他 [`~modular_pipelines.ModularPipelineBlocks`] 以循环方式组合在一起。数据循环流动，使用 `intermediate_inputs` 和 `intermediate_outputs`，并且每个块都是迭代运行的。这通常用于创建一个默认是迭代的去噪循环。

本指南向您展示如何创建 [`~modular_pipelines.LoopSequentialPipelineBlocks`]。

## 循环包装器

[`~modular_pipelines.LoopSequentialPipelineBlocks`]，也被称为 *循环包装器*，因为它定义了循环结构、迭代变量和配置。在循环包装器内，您需要以下变量。

- `loop_inputs` 是用户提供的值，等同于 [`~modular_pipelines.ModularPipelineBlocks.inputs`]。
- `loop_intermediate_inputs` 是来自 [`~modular_pipelines.PipelineState`] 的中间变量，等同于 [`~modular_pipelines.ModularPipelineBlocks.intermediate_inputs`]。
- `loop_intermediate_outputs` 是由块创建并添加到 [`~modular_pipelines.PipelineState`] 的新中间变量。它等同于 [`~modular_pipelines.ModularPipelineBlocks.intermediate_outputs`]。
- `__call__` 方法定义了循环结构和迭代逻辑。

```py
import torch
from diffusers.modular_pipelines import LoopSequentialPipelineBlocks, ModularPipelineBlocks, InputParam, OutputParam

class LoopWrapper(LoopSequentialPipelineBlocks):
    model_name = "test"
    @property
    def description(self):
        return "I'm a loop!!"
    @property
    def loop_inputs(self):
        return [InputParam(name="num_steps")]
    @torch.no_grad()
    def __call__(self, components, state):
        block_state = self.get_block_state(state)
        # 循环结构 - 可以根据您的需求定制
        for i in range(block_state.num_steps):
            # loop_step 按顺序执行所有注册的块
            components, block_state = self.loop_step(components, block_state, i=i)
        self.set_block_state(state, block_state)
        return components, state
```

循环包装器可以传递额外的参数，如当前迭代索引，到循环块。

## 循环块

循环块是一个 [`~modular_pipelines.ModularPipelineBlocks`]，但 `__call__` 方法的行为不同。

- 它从循环包装器。
- 它直接与[`~modular_pipelines.BlockState`]一起工作，而不是[`~modular_pipelines.PipelineState`]。
- 它不需要检索或更新[`~modular_pipelines.BlockState`]。

循环块共享相同的[`~modular_pipelines.BlockState`]，以允许值在循环的每次迭代中累积和变化。

```py
class LoopBlock(ModularPipelineBlocks):
    model_name = "test"
    @property
    def inputs(self):
        return [InputParam(name="x")]
    @property
    def intermediate_outputs(self):
        # 这个块产生的输出
        return [OutputParam(name="x")]
    @property
    def description(self):
        return "我是一个在`LoopWrapper`类内部使用的块"
    def __call__(self, components, block_state, i: int):
        block_state.x += 1
        return components, block_state
```

## LoopSequentialPipelineBlocks

使用[`~modular_pipelines.LoopSequentialPipelineBlocks.from_blocks_dict`]方法将循环块添加到循环包装器中，以创建[`~modular_pipelines.LoopSequentialPipelineBlocks`]。

```py
loop = LoopWrapper.from_blocks_dict({"block1": LoopBlock})
```

添加更多的循环块以在每次迭代中运行，使用[`~modular_pipelines.LoopSequentialPipelineBlocks.from_blocks_dict`]。这允许您在不改变循环逻辑本身的情况下修改块。

```py
loop = LoopWrapper.from_blocks_dict({"block1": LoopBlock(), "block2": LoopBlock})
```
