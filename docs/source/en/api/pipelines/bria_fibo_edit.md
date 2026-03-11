<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Bria Fibo Edit

Fibo Edit is an 8B parameter image-to-image model that introduces a new paradigm of structured control, operating on JSON inputs paired with source images to enable deterministic and repeatable editing workflows.
Featuring native masking for granular precision, it moves beyond simple prompt-based diffusion to offer explicit, interpretable control optimized for production environments.
Its lightweight architecture is designed for deep customization, empowering researchers to build specialized "Edit" models for domain-specific tasks while delivering top-tier aesthetic quality

## Usage
_As the model is gated, before using it with diffusers you first need to go to the [Bria Fibo Hugging Face page](https://huggingface.co/briaai/Fibo-Edit), fill in the form and accept the gate. Once you are in, you need to login so that your system knows you’ve accepted the gate._

Use the command below to log in:

```bash
hf auth login
```


## BriaFiboEditPipeline

[[autodoc]] BriaFiboEditPipeline
	- all
	- __call__