<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Reduce memory usage

Modern diffusion models like [Flux](../api/pipelines/flux) and [Wan](../api/pipelines/wan) have billions of parameters that take up a lot of memory on your hardware for inference. This poses a challenge because common GPUs often don't have sufficient memory.

To overcome these memory constraints, you can use a second GPU (if available), offload some of the pipeline components to the CPU, and more. This guide will show you how to reduce your memory usage.

## Multiple GPUs

If you have access to more than one GPU, there a few options for efficiently loading and distributing a large model across your hardware. These features are supported by the [Accelerate](https://huggingface.co/docs/accelerate/index) library, so make sure it is installed first.

```bash
pip install -U accelerate
```

### Sharded checkpoints



### Device placement

### Sliced VAE

## Tiled VAE

## CPU offloading

## Model offloading

## Group offloading

## FP8 layerwise weight-casting

## Channels-last format

## Tracing

## Memory efficient attention
