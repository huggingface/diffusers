<!--版权 2025 The HuggingFace Team。保留所有权利。

根据Apache许可证2.0版（"许可证"）授权；除非符合许可证的规定，否则不得使用此文件。
您可以在以下网址获取许可证的副本

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据许可证分发的软件是基于"按原样"分发的，没有任何形式的明示或暗示的担保或条件。有关许可证下特定的语言管理权限和限制，请参阅许可证。
-->

# 状态

块依赖于[`~modular_pipelines.PipelineState`]和[`~modular_pipelines.BlockState`]数据结构进行通信和数据共享。

| 状态 | 描述 |
|-------|-------------|
| [`~modular_pipelines.PipelineState`] | 维护管道执行所需的整体数据，并允许块读取和更新其数据。 |
| [`~modular_pipelines.BlockState`] | 允许每个块使用来自`inputs`的必要数据执行其计算 |

本指南解释了状态如何工作以及它们如何连接块。

## PipelineState

[`~modular_pipelines.PipelineState`]是所有块的全局状态容器。它维护管道的完整运行时状态，并为块提供了一种结构化的方式来读取和写入共享数据。

[`~modular_pipelines.PipelineState`]中有两个字典用于结构化数据。

- `values`字典是一个**可变**状态，包含用户提供的输入值的副本和由块生成的中间输出值。如果一个块修改了一个`input`，它将在调用`set_block_state`后反映在`values`字典中。

```py
PipelineState(
  values={
    'prompt': 'a cat'
    'guidance_scale': 7.0
    'num_inference_steps': 25
    'prompt_embeds': Tensor(dtype=torch.float32, shape=torch.Size([1, 1, 1, 1]))
    'negative_prompt_embeds': None
  },
)
```

## BlockState

[`~modular_pipelines.BlockState`]是[`~modular_pipelines.PipelineState`]中相关变量的局部视图，单个块需要这些变量来执行其计算。

直接作为属性访问这些变量，如`block_state.image`。

```py
BlockState(
    image: <PIL.Image.Image image mode=RGB size=512x512 at 0x7F3ECC494640>
)
```

当一个块的`__call__`方法被执行时，它用`self.get_block_state(state)`检索[`BlockState`]，执行其操作，并用`self.set_block_state(state, block_state)`更新[`~modular_pipelines.PipelineState`]。

```py
def __call__(self, components, state):
    # 检索BlockState
    block_state = self.get_block_state(state)

    # 对输入进行计算的逻辑

    # 更新PipelineState
    self.set_block_state(state, block_state)
    return components, state
```

## 状态交互

[`~modular_pipelines.PipelineState`]和[`~modular_pipelines.BlockState`]的交互由块的`inputs`和`intermediate_outputs`定义。

- `inputs`,
一个块可以修改输入 - 比如 `block_state.image` - 并且这个改变可以通过调用 `set_block_state` 全局传播到 [`~modular_pipelines.PipelineState`]。
- `intermediate_outputs`，是一个块创建的新变量。它被添加到 [`~modular_pipelines.PipelineState`] 的 `values` 字典中，并且可以作为后续块的可用变量，或者由用户作为管道的最终输出访问。
