<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Overview

A pipeline is an end-to-end class that provides a quick and easy way to use a diffusion system for inference by bundling independently trained models and schedulers together. Certain combinations of models and schedulers define specific pipeline types, like [`StableDiffusionXLPipeline`] or [`StableDiffusionControlNetPipeline`], with specific capabilities. All pipeline types inherit from the base [`DiffusionPipeline`] class; pass it any checkpoint, and it'll automatically detect the pipeline type and load the necessary components.

This section demonstrates how to use specific pipelines such as Stable Diffusion XL, ControlNet, and DiffEdit. You'll also learn how to use a distilled version of the Stable Diffusion model to speed up inference, how to create reproducible pipelines, and how to use and contribute community pipelines.
