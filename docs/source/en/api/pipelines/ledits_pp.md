<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# LEDITS++

LEDITS++ was proposed in [LEDITS++: Limitless Image Editing using Text-to-Image Models](https://huggingface.co/papers/2311.16711) by Manuel Brack, Felix Friedrich, Katharina Kornmeier, Linoy Tsaban, Patrick Schramowski, Kristian Kersting, Apolinário Passos.

The abstract from the paper is:

*Text-to-image diffusion models have recently received increasing interest for their astonishing ability to produce high-fidelity images from solely text inputs. Subsequent research efforts aim to exploit and apply their capabilities to real image editing. However, existing image-to-image methods are often inefficient, imprecise, and of limited versatility. They either require time-consuming fine-tuning, deviate unnecessarily strongly from the input image, and/or lack support for multiple, simultaneous edits. To address these issues, we introduce LEDITS++, an efficient yet versatile and precise textual image manipulation technique. LEDITS++'s novel inversion approach requires no tuning nor optimization and produces high-fidelity results with a few diffusion steps. Second, our methodology supports multiple simultaneous edits and is architecture-agnostic. Third, we use a novel implicit masking technique that limits changes to relevant image regions. We propose the novel TEdBench++ benchmark as part of our exhaustive evaluation. Our results demonstrate the capabilities of LEDITS++ and its improvements over previous methods. The project page is available at https://leditsplusplus-project.static.hf.space .*

<Tip>

You can find additional information about LEDITS++ on the [project page](https://leditsplusplus-project.static.hf.space/index.html) and try it out in a [demo](https://huggingface.co/spaces/editing-images/leditsplusplus).

</Tip>

<Tip warning={true}>
Due to some backward compatability issues with the current diffusers implementation of [`~schedulers.DPMSolverMultistepScheduler`] this implementation of LEdits++ can no longer guarantee perfect inversion.
This issue is unlikely to have any noticeable effects on applied use-cases. However, we provide an alternative implementation that guarantees perfect inversion in a dedicated [GitHub repo](https://github.com/ml-research/ledits_pp).
</Tip>

We provide two distinct pipelines based on different pre-trained models.

## LEditsPPPipelineStableDiffusion
[[autodoc]] pipelines.ledits_pp.LEditsPPPipelineStableDiffusion
	- all
	- __call__
	- invert

## LEditsPPPipelineStableDiffusionXL
[[autodoc]] pipelines.ledits_pp.LEditsPPPipelineStableDiffusionXL
	- all
	- __call__
	- invert



## LEditsPPDiffusionPipelineOutput
[[autodoc]] pipelines.ledits_pp.pipeline_output.LEditsPPDiffusionPipelineOutput
	- all

## LEditsPPInversionPipelineOutput
[[autodoc]] pipelines.ledits_pp.pipeline_output.LEditsPPInversionPipelineOutput
	- all
