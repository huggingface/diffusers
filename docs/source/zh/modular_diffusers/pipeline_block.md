<!--版权 2025 The HuggingFace Team。保留所有权利。

根据Apache许可证2.0版（“许可证”）授权；除非符合许可证，否则不得使用此文件。您可以在

http://www.apache.org/licenses/LICENSE-2.0

获取许可证的副本。

除非适用法律要求或书面同意，根据许可证分发的软件是基于“按原样”基础分发的，没有任何明示或暗示的保证或条件。请参阅许可证了解特定语言管理权限和限制。
-->

# ModularPipelineBlocks

[`~modular_pipelines.ModularPipelineBlocks`] 是构建 [`ModularPipeline`] 的基本块。它定义了管道中特定步骤应执行的组件、输入/输出和计算。一个 [`~modular_pipelines.ModularPipelineBlocks`] 与其他块连接，使用 [状态](./modular_diffusers_states)，以实现工作流的模块化构建。

单独的 [`~modular_pipelines.ModularPipelineBlocks`] 无法执行。它是管道中步骤应执行的操作的蓝图。要实际运行和执行管道，需要将 [`~modular_pipelines.ModularPipelineBlocks`] 转换为 [`ModularPipeline`]。

本指南将向您展示如何创建 [`~modular_pipelines.ModularPipelineBlocks`]。

## 输入和输出

> [!TIP]
> 如果您不熟悉Modular Diffusers中状态的工作原理，请参考 [States](./modular_diffusers_states) 指南。

一个 [`~modular_pipelines.ModularPipelineBlocks`] 需要 `inputs` 和 `intermediate_outputs`。

- `inputs` 是由用户提供并从 [`~modular_pipelines.PipelineState`] 中检索的值。这很有用，因为某些工作流会调整图像大小，但仍需要原始图像。 [`~modular_pipelines.PipelineState`] 维护原始图像。

    使用 `InputParam` 定义 `inputs`。

    ```py
    from diffusers.modular_pipelines import InputParam

    user_inputs = [
        InputParam(name="image", type_hint="PIL.Image", description="要处理的原始输入图像")
    ]
    ```

- `intermediate_inputs` 通常由前一个块创建的值，但如果前面的块没有生成它们，也可以直接提供。与 `inputs` 不同，`intermediate_inputs` 可以被修改。

    使用 `InputParam` 定义 `intermediate_inputs`。

    ```py
    user_intermediate_inputs = [
        InputParam(name="processed_image", type_hint="torch.Tensor", description="image that has been preprocessed and normalized"),
    ]
    ```

- `intermediate_outputs` 是由块创建并添加到 [`~modular_pipelines.PipelineState`] 的新值。`intermediate_outputs` 可作为后续块的 `intermediate_inputs` 使用，或作为运行管道的最终输出使用。

    使用 `OutputParam` 定义 `intermediate_outputs`。

    ```py
    from diffusers.modular_pipelines import OutputParam

        user_intermediate_outputs = [
        OutputParam(name="image_latents", description="latents representing the image")
    ]
    ```

中间输入和输出共享数据以连接块。它们可以在任何时候访问，允许你跟踪工作流的进度。

## 计算逻辑

一个块执行的计算在`__call__`方法中定义，它遵循特定的结构。

1. 检索[`~modular_pipelines.BlockState`]以获取`inputs`和`intermediate_inputs`的局部视图。
2. 在`inputs`和`intermediate_inputs`上实现计算逻辑。
3. 更新[`~modular_pipelines.PipelineState`]以将局部[`~modular_pipelines.BlockState`]的更改推送回全局[`~modular_pipelines.PipelineState`]。
4. 返回对下一个块可用的组件和状态。

```py
def __call__(self, components, state):
    # 获取该块需要的状态变量的局部视图
    block_state = self.get_block_state(state)

    # 你的计算逻辑在这里
    # block_state包含你所有的inputs和intermediate_inputs
    # 像这样访问它们: block_state.image, block_state.processed_image

    # 用你更新的block_states更新管道状态
    self.set_block_state(state, block_state)
    return components, state
```

### 组件和配置

块需要的组件和管道级别的配置在[`ComponentSpec`]和[`~modular_pipelines.ConfigSpec`]中指定。

- [`ComponentSpec`]包含块使用的预期组件。你需要组件的`name`和理想情况下指定组件确切是什么的`type_hint`。
- [`~modular_pipelines.ConfigSpec`]包含控制所有块行为的管道级别设置。

```py
from diffusers import ComponentSpec, ConfigSpec

expected_components = [
    ComponentSpec(name="unet", type_hint=UNet2DConditionModel),
    ComponentSpec(name="scheduler", type_hint=EulerDiscreteScheduler)
]

expected_config = [
    ConfigSpec("force_zeros_for_empty_prompt", True)
]
```

当块被转换为管道时，组件作为`__call__`中的第一个参数对块可用。

```py
def __call__(self, components, state):
    # 使用点符号访问组件
    unet = components.unet
    vae = components.vae
    scheduler = components.scheduler
```
