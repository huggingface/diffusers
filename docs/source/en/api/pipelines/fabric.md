<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# FabricPipeline

[FABRIC: Personalizing Diffusion Models with Iterative Feedback](https://huggingface.co/papers/2307.10159) (FABRIC) is by Dimitri von Rütte, Elisabetta Fedele, Jonathan Thomm and Lukas Wolf

FABRIC is training-free approach that conditions the diffusion process on a set of feedback images, applicable to a wide range of popular diffusion models, created by the researchers and engineers from [ETH Zürich, Switzerland](https://github.com/sd-fabric). The [`FabricPipeline`] is capable of generating photo-realistic images given any text input using Stable Diffusion and finetune them on the basis of feedback.

The abstract of the paper is the following:

*In an era where visual content generation is increasingly driven by machine learning, the integration of human feedback into generative models presents significant opportunities for enhancing user experience and output quality. This study explores strategies for incorporating iterative human feedback into the generative process of diffusion-based text-to-image models. We propose FABRIC, a training-free approach applicable to a wide range of popular diffusion models, which exploits the self-attention layer present in the most widely used architectures to condition the diffusion process on a set of feedback images. To ensure a rigorous assessment of our approach, we introduce a comprehensive evaluation methodology, offering a robust mechanism to quantify the performance of generative visual models that integrate human feedback. We show that generation results improve over multiple rounds of iterative feedback through exhaustive analysis, implicitly optimizing arbitrary user preferences. The potential applications of these findings extend to fields such as personalized content creation and customization*

The original codebase can be found here: 
- *FABRIC*: [sd-fabric/fabric](https://github.com/sd-fabric/fabric)

Available Checkpoints are:
- *dreamlike-photoreal-2.0* [dreamlike-art/dreamlike-photoreal-2.0](https://huggingface.co/dreamlike-art/dreamlike-photoreal-2.0)


## Usage Example

Before using Fabric make sure to have `transformers`, `accelerate`, `huggingface_hub`  installed. 
You can install the libraries as follows:

```
pip install transformers
pip install accelerate
pip install huggingface_hub
```

### Text-to-Image

You can use Fabric as follows for *text-to-image*:

```py
from diffusers import FabricPipeline
import torch

model_id = "dreamlike-art/dreamlike-photoreal-2.0"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
neg_prompt = "bad anatomy, cropped, lowres"
image = pipe(prompt=prompt, negative_prompt=neg_prompt).images[0]
```

You can use Fabric as follows for *text-to-image-with-feedback*:

```py
from diffusers import FabricPipeline
import torch

model_id = "dreamlike-art/dreamlike-photoreal-2.0"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
neg_prompt = "bad anatomy, cropped, lowres"
liked = ["path/to/image"]
disliked = ["path/to/image"]
image = pipe(prompt=prompt, negative_prompt=neg_prompt,liked=liked,disliked=disliked).images[0]
```

Let's have a look at the images (*512X512*)

| Without Feedback            | With Feedback  (1st image)          |
|---------------------|---------------------|
| ![Image 1](https://drive.google.com/uc?export=view&id=12wxbikt7834eRTK40legR5PtJmFLNH34) | ![Feedback Image 1](https://drive.google.com/uc?export=view&id=1YcFPDHSRr2OE3hy-5lvr8An21Jum85D5) | 


[[autodoc]] FabricPipeline
	- all
	- __call__
