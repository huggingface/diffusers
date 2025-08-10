<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Bria 3.2

Bria 3.2 is the next-generation commercial-ready text-to-image model. With just 4 billion parameters, it provides exceptional aesthetics and text rendering, evaluated to provide on par results to leading open-source models, and outperforming other licensed models.
In addition to being built entirely on licensed data, 3.2 provides several advantages for enterprise and commercial use:

- Efficient Compute - the model is X3 smaller than the equivalent models in the market (4B parameters vs 12B parameters other open source models)
- Architecture Consistency: Same architecture as 3.1—ideal for users looking to upgrade without disruption.
- Fine-tuning Speedup: 2x faster fine-tuning on L40S and A100.

Original model checkpoints for Bria 3.2 can be found [here](https://huggingface.co/briaai/BRIA-3.2).
Github repo for Bria 3.2 can be found [here](https://github.com/Bria-AI/BRIA-3.2).

If you want to learn more about the Bria platform, and get free traril access, please visit [bria.ai](https://bria.ai).


## Usage

_As the model is gated, before using it with diffusers you first need to go to the [Bria 3.2 Hugging Face page](https://huggingface.co/briaai/BRIA-3.2), fill in the form and accept the gate. Once you are in, you need to login so that your system knows you’ve accepted the gate._

Use the command below to log in:

```bash
hf auth login
```


## BriaPipeline

[[autodoc]] BriaPipeline
	- all
	- __call__

