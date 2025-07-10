<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Getting Started with Modular Diffusers

<Tip warning={true}>

🧪 **Experimental Feature**: Modular Diffusers is an experimental feature we are actively developing. The API may be subject to breaking changes.

</Tip>

With Modular Diffusers, we introduce a unified pipeline system that simplifies how you work with diffusion models. Instead of creating separate pipelines for each task, Modular Diffusers lets you:

**Write Only What's New**: You won't need to write an entire pipeline from scratch every time you have a new use case. You can create pipeline blocks just for your new workflow's unique aspects and reuse existing blocks for existing functionalities. 

**Assemble Like LEGO®**: You can mix and match between blocks in flexible ways. This allows you to write dedicated blocks unique to specific workflows, and then assemble different blocks into a pipeline that can be used more conveniently for multiple workflows. 


Here's how our guides are organized to help you navigate the Modular Diffusers documentation:

### 🚀 Running Pipelines
- **[Modular Pipeline Guide](./modular_pipeline.md)** - How to use predefined blocks to build a pipeline and run it
- **[Components Manager Guide](./components_manager.md)** - How to manage and reuse components across multiple pipelines

### 📚 Creating PipelineBlocks
- **[Pipeline and Block States](./modular_diffusers_states.md)** - Understanding PipelineState and BlockState
- **[Pipeline Block](./pipeline_block.md)** - How to write custom PipelineBlocks
- **[SequentialPipelineBlocks](sequential_pipeline_blocks.md)** - Connecting blocks in sequence
- **[LoopSequentialPipelineBlocks](./loop_sequential_pipeline_blocks.md)** - Creating iterative workflows
- **[AutoPipelineBlocks](./auto_pipeline_blocks.md)** - Conditional block selection

### 🎯 Practical Examples
- **[End-to-End Example](./end_to_end_guide.md)** - Complete end-to-end examples including sharing your workflow in huggingface hub and deplying UI nodes
