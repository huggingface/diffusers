<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# AutoPipeline

The `AutoPipeline` is designed to make it easy to load a checkpoint for a task without needing to know the specific pipeline class. Based on the task, the `AutoPipeline` automatically retrieves the correct pipeline class from the checkpoint `model_index.json` file.

> [!TIP]
> Check out the [AutoPipeline](../../tutorials/autopipeline) tutorial to learn how to use this API!

## AutoPipelineForText2Image

[[autodoc]] AutoPipelineForText2Image
	- all
	- from_pretrained
	- from_pipe

## AutoPipelineForImage2Image

[[autodoc]] AutoPipelineForImage2Image
	- all
	- from_pretrained
	- from_pipe

## AutoPipelineForInpainting

[[autodoc]] AutoPipelineForInpainting
	- all
	- from_pretrained
	- from_pipe
