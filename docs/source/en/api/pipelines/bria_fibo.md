<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Bria Fibo

Text-to-image models have mastered imagination - but not control. FIBO changes that.

FIBO is trained on structured JSON captions up to 1,000+ words and designed to understand and control different visual parameters such as lighting, composition, color, and camera settings, enabling precise and reproducible outputs.

With only 8 billion parameters, FIBO provides a new level of image quality, prompt adherence and proffesional control.

## Usage

_As the model is gated, before using it with diffusers you first need to go to the [Bria Fibo Hugging Face page](https://huggingface.co/briaai/FIBO), fill in the form and accept the gate. Once you are in, you need to login so that your system knows you’ve accepted the gate._

Use the command below to log in:

```bash
hf auth login
```


## BriaPipeline

[[autodoc]] BriaPipeline
	- all
	- __call__

