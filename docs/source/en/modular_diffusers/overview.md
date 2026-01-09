<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Overview

> [!WARNING]
> Modular Diffusers is under active development and it's API may change.

Modular Diffusers is a unified pipeline system that simplifies your workflow with *pipeline blocks*.

- Blocks are reusable and you only need to create new blocks that are unique to your pipeline.
- Blocks can be mixed and matched to adapt to or create a pipeline for a specific workflow or multiple workflows.

The Modular Diffusers docs are organized as shown below.

## Quickstart

- A [quickstart](./quickstart) demonstrating how to implement an example workflow with Modular Diffusers.

## ModularPipelineBlocks

- [States](./modular_diffusers_states) explains how data is shared and communicated between blocks and [`ModularPipeline`].
- [ModularPipelineBlocks](./pipeline_block) is the most basic unit of a [`ModularPipeline`] and this guide shows you how to create one.
- [SequentialPipelineBlocks](./sequential_pipeline_blocks) is a type of block that chains multiple blocks so they run one after another, passing data along the chain. This guide shows you how to create [`~modular_pipelines.SequentialPipelineBlocks`] and how they connect and work together.
- [LoopSequentialPipelineBlocks](./loop_sequential_pipeline_blocks) is a type of block that runs a series of blocks in a loop. This guide shows you how to create [`~modular_pipelines.LoopSequentialPipelineBlocks`].
- [AutoPipelineBlocks](./auto_pipeline_blocks) is a type of block that automatically chooses which blocks to run based on the input. This guide shows you how to create [`~modular_pipelines.AutoPipelineBlocks`].

## ModularPipeline

- [ModularPipeline](./modular_pipeline) shows you how to create and convert pipeline blocks into an executable [`ModularPipeline`].
- [ComponentsManager](./components_manager) shows you how to manage and reuse components across multiple pipelines.
- [Guiders](./guiders) shows you how to use different guidance methods in the pipeline.