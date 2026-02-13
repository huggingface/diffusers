<!--版权所有 2025 The HuggingFace Team。保留所有权利。

根据Apache许可证2.0版（"许可证"）授权；除非符合许可证，否则不得使用此文件。您可以在以下位置获取许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据许可证分发的软件按"原样"分发，无任何明示或暗示的担保或条件。有关许可证下特定语言的权限和限制，请参阅许可证。
-->

# 概述

> [!WARNING]
> 模块化Diffusers正在积极开发中，其API可能会发生变化。

模块化Diffusers是一个统一的管道系统，通过*管道块*简化您的工作流程。

- 块是可重用的，您只需要为您的管道创建独特的块。
- 块可以混合搭配，以适应或为特定工作流程或多个工作流程创建管道。

模块化Diffusers文档的组织如下所示。

## 快速开始

- 一个[快速开始](./quickstart)演示了如何使用模块化Diffusers实现一个示例工作流程。

## ModularPipelineBlocks

- [States](./modular_diffusers_states)解释了数据如何在块和[`ModularPipeline`]之间共享和通信。
- [ModularPipelineBlocks](./pipeline_block)是[`ModularPipeline`]最基本的单位，本指南向您展示如何创建一个。
- [SequentialPipelineBlocks](./sequential_pipeline_blocks)是一种类型的块，它将多个块链接起来，使它们一个接一个地运行，沿着链传递数据。本指南向您展示如何创建[`~modular_pipelines.SequentialPipelineBlocks`]以及它们如何连接和一起工作。
- [LoopSequentialPipelineBlocks](./loop_sequential_pipeline_blocks)是一种类型的块，它在循环中运行一系列块。本指南向您展示如何创建[`~modular_pipelines.LoopSequentialPipelineBlocks`]。
- [AutoPipelineBlocks](./auto_pipeline_blocks)是一种类型的块，它根据输入自动选择要运行的块。本指南向您展示如何创建[`~modular_pipelines.AutoPipelineBlocks`]。

## ModularPipeline

- [ModularPipeline](./modular_pipeline)向您展示如何创建并将管道块转换为可执行的[`ModularPipeline`]。
- [ComponentsManager](./components_manager)向您展示如何跨多个管道管理和重用组件。
- [Guiders](./guiders)向您展示如何在管道中使用不同的指导方法。
