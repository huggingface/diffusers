<!--版权 2025 HuggingFace 团队。保留所有权利。

根据 Apache 许可证 2.0 版本（"许可证"）授权；
除非符合许可证要求，否则不得使用本文件。
您可以在以下网址获取许可证副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，本软件按"原样"分发，
无任何明示或暗示的担保或条件。详见许可证中
的特定语言规定和限制。
-->

# 设计哲学

🧨 Diffusers 提供**最先进**的预训练扩散模型支持多模态任务。
其目标是成为推理和训练通用的**模块化工具箱**。

我们致力于构建一个经得起时间考验的库，因此对API设计极为重视。

简而言之，Diffusers 被设计为 PyTorch 的自然延伸。因此，我们的多数设计决策都基于 [PyTorch 设计原则](https://pytorch.org/docs/stable/community/design.html#pytorch-design-philosophy)。以下是核心原则：

## 可用性优先于性能

- 尽管 Diffusers 包含众多性能优化特性（参见[内存与速度优化](https://huggingface.co/docs/diffusers/optimization/fp16)），模型默认总是以最高精度和最低优化级别加载。因此除非用户指定，扩散流程(pipeline)默认在CPU上以float32精度初始化。这确保了跨平台和加速器的可用性，意味着运行本库无需复杂安装。
- Diffusers 追求**轻量化**，仅有少量必需依赖，但提供诸多可选依赖以提升性能（如`accelerate`、`safetensors`、`onnx`等）。我们竭力保持库的轻量级特性，使其能轻松作为其他包的依赖项。
- Diffusers 偏好简单、自解释的代码而非浓缩的"魔法"代码。这意味着lambda函数等简写语法和高级PyTorch操作符通常不被采用。

## 简洁优于简易

正如PyTorch所言：**显式优于隐式**，**简洁优于复杂**。这一哲学体现在库的多个方面：
- 我们遵循PyTorch的API设计，例如使用[`DiffusionPipeline.to`](https://huggingface.co/docs/diffusers/main/en/api/diffusion_pipeline#diffusers.DiffusionPipeline.to)让用户自主管理设备。
- 明确的错误提示优于静默纠正错误输入。Diffusers 旨在教育用户，而非单纯降低使用难度。
- 暴露复杂的模型与调度器(scheduler)交互逻辑而非内部魔法处理。调度器/采样器与扩散模型分离且相互依赖最小化，迫使用户编写展开的去噪循环。但这种分离便于调试，并赋予用户更多控制权来调整去噪过程或切换模型/调度器。
- 扩散流程中独立训练的组件（如文本编码器、UNet、变分自编码器）各有专属模型类。这要求用户处理组件间交互，且序列化格式将组件分存不同文件。但此举便于调试和定制，得益于组件分离，DreamBooth或Textual Inversion训练变得极为简单。

## 可定制与贡献友好优于抽象

库的大部分沿用了[Transformers库](https://github.com/huggingface/transformers)的重要设计原则：宁要重复代码，勿要仓促抽象。这一原则与[DRY原则](https://en.wikipedia.org/wiki/Don%27t_repeat_yourself)形成鲜明对比。

简言之，正如Transformers对建模文件的做法，Diffusers对流程(pipeline)和调度器(scheduler)保持极低抽象度与高度自包含代码。函数、长代码块甚至类可能在多文件中重复，初看像是糟糕的松散设计。但该设计已被Transformers证明极其成功，对社区驱动的开源机器学习库意义重大：
- 机器学习领域发展迅猛，范式、模型架构和算法快速迭代，难以定义长效代码抽象。
- ML从业者常需快速修改现有代码进行研究，因此偏好自包含代码而非多重抽象。
- 开源库依赖社区贡献，必须构建易于参与的代码库。抽象度越高、依赖越复杂、可读性越差，贡献难度越大。过度抽象的库会吓退贡献者。若贡献不会破坏核心功能，不仅吸引新贡献者，也更便于并行审查和修改。

Hugging Face称此设计为**单文件政策**——即某个类的几乎所有代码都应写在单一自包含文件中。更多哲学探讨可参阅[此博文](https://huggingface.co/blog/transformers-design-philosophy)。

Diffusers对流程和调度器完全遵循该哲学，但对diffusion模型仅部分适用。原因在于多数扩散流程（如[DDPM](https://huggingface.co/docs/diffusers/api/pipelines/ddpm)、[Stable Diffusion](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/overview#stable-diffusion-pipelines)、[unCLIP (DALL·E 2)](https://huggingface.co/docs/diffusers/api/pipelines/unclip)和[Imagen](https://imagen.research.google/)）都基于相同扩散模型——[UNet](https://huggingface.co/docs/diffusers/api/models/unet2d-cond)。

现在您应已理解🧨 Diffusers的设计理念🤗。我们力求在全库贯彻这些原则，但仍存在少数例外或欠佳设计。如有反馈，我们❤️欢迎在[GitHub提交](https://github.com/huggingface/diffusers/issues/new?assignees=&labels=&template=feedback.md&title=)。

## 设计哲学细节

现在深入探讨设计细节。Diffusers主要包含三类：[流程(pipeline)](https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines)、[模型](https://github.com/huggingface/diffusers/tree/main/src/diffusers/models)和[调度器(scheduler)](https://github.com/huggingface/diffusers/tree/main/src/diffusers/schedulers)。以下是各类的具体设计决策。

### 流程(Pipelines)

流程设计追求易用性（因此不完全遵循[*简洁优于简易*](#简洁优于简易)），不要求功能完备，应视为使用[模型](#模型)和[调度器](#调度器schedulers)进行推理的示例。

遵循原则：
- 采用单文件政策。所有流程位于src/diffusers/pipelines下的独立目录。一个流程文件夹对应一篇扩散论文/项目/发布。如[`src/diffusers/pipelines/stable-diffusion`](https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines/stable_diffusion)可包含多个流程文件。若流程功能相似，可使用[# Copied from机制](https://github.com/huggingface/diffusers/blob/125d783076e5bd9785beb05367a2d2566843a271/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_img2img.py#L251)。
- 所有流程继承[`DiffusionPipeline`]。
- 每个流程由不同模型和调度器组件构成，这些组件记录于[`model_index.json`文件](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/blob/main/model_index.json)，可通过同名属性访问，并可用[`DiffusionPipeline.components`](https://huggingface.co/docs/diffusers/main/en/api/diffusion_pipeline#diffusers.DiffusionPipeline.components)在流程间共享。
- 所有流程应能通过[`DiffusionPipeline.from_pretrained`](https://huggingface.co/docs/diffusers/main/en/api/diffusion_pipeline#diffusers.DiffusionPipeline.from_pretrained)加载。
- 流程**仅**用于推理。
- 流程代码应具备高可读性、自解释性和易修改性。
- 流程应设计为可相互构建，便于集成到高层API。
- 流程**非**功能完备的用户界面。完整UI推荐[InvokeAI](https://github.com/invoke-ai/InvokeAI)、[Diffuzers](https://github.com/abhishekkrthakur/diffuzers)或[lama-cleaner](https://github.com/Sanster/lama-cleaner)。
- 每个流程应通过唯一的`__call__`方法运行，且参数命名应跨流程统一。
- 流程应以其解决的任务命名。
- 几乎所有新diffusion流程都应在新文件夹/文件中实现。

### 模型

模型设计为可配置的工具箱，是[PyTorch Module类](https://pytorch.org/docs/stable/generated/torch.nn.Module.html)的自然延伸，仅部分遵循**单文件政策**。

遵循原则：
- 模型对应**特定架构类型**。如[`UNet2DConditionModel`]类适用于所有需要2D图像输入且受上下文调节的UNet变体。
- 所有模型位于[`src/diffusers/models`](https://github.com/huggingface/diffusers/tree/main/src/diffusers/models)，每种架构应有独立文件，如[`unets/unet_2d_condition.py`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/unets/unet_2d_condition.py)、[`transformers/transformer_2d.py`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_2d.py)等。
- 模型**不**采用单文件政策，应使用小型建模模块如[`attention.py`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention.py)、[`resnet.py`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/resnet.py)、[`embeddings.py`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/embeddings.py)等。**注意**：这与Transformers的建模文件截然不同，表明模型未完全遵循单文件政策。
- 模型意图暴露复杂度（类似PyTorch的`Module`类），并提供明确错误提示。
- 所有模型继承`ModelMixin`和`ConfigMixin`。
- 当不涉及重大代码变更、保持向后兼容性且显著提升内存/计算效率时，可对模型进行性能优化。
- 模型默认应具备最高精度和最低性能设置。
- 若新模型检查点可归类为现有架构，应适配现有架构而非新建文件。仅当架构根本性不同时才创建新文件。
- 模型设计应便于未来扩展。可通过限制公开函数参数、配置参数和"预见"变更实现。例如：优先采用可扩展的`string`类型参数而非布尔型`is_..._type`参数。对现有架构的修改应保持最小化。
- 模型设计需在代码可读性与多检查点支持间权衡。多数情况下应适配现有类，但某些例外（如[UNet块](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/unets/unet_2d_blocks.py)和[注意力处理器](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py)）需新建类以保证长期可读性。

### 调度器(Schedulers)

调度器负责引导推理去噪过程及定义训练噪声计划。它们设计为独立的可加载配置类，严格遵循**单文件政策**。

遵循原则：
- 所有调度器位于[`src/diffusers/schedulers`](https://github.com/huggingface/diffusers/tree/main/src/diffusers/schedulers)。
- 调度器**禁止**从大型工具文件导入，必须保持高度自包含。
- 一个调度器Python文件对应一种算法（如论文定义的算法）。
- 若调度器功能相似，可使用`# Copied from`机制。
- 所有调度器继承`SchedulerMixin`和`ConfigMixin`。
- 调度器可通过[`ConfigMixin.from_config`](https://huggingface.co/docs/diffusers/main/en/api/configuration#diffusers.ConfigMixin.from_config)轻松切换（详见[此处](../using-diffusers/schedulers)）。
- 每个调度器必须包含`set_num_inference_steps`和`step`函数。在每次去噪过程前（即调用`step(...)`前）必须调用`set_num_inference_steps(...)`。
- 每个调度器通过`timesteps`属性暴露需要"循环"的时间步，这是模型将被调用的时间步数组。
- `step(...)`函数接收模型预测输出和"当前"样本(x_t)，返回"前一个"略去噪的样本(x_t-1)。
- 鉴于扩散调度器的复杂性，`step`函数不暴露全部细节，可视为"黑盒"。
- 几乎所有新调度器都应在新文件中实现。