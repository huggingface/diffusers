<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Prompt-to-Prompt

[Prompt-to-Prompt Image Editing with Cross Attention Control](https://huggingface.co/papers/2208.01626) is by Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch and Daniel Cohen-Or.

The abstract from the paper is:

*Recent large-scale text-driven synthesis models have attracted much attention thanks to their remarkable capabilities of generating highly diverse images that follow given text prompts. Such text-based synthesis methods are particularly appealing to humans who are used to verbally describe their intent. Therefore, it is only natural to extend the text-driven image synthesis to text-driven image editing. Editing is challenging for these generative models, since an innate property of an editing technique is to preserve most of the original image, while in the text-based models, even a small modification of the text prompt often leads to a completely different outcome. State-of-the-art methods mitigate this by requiring the users to provide a spatial mask to localize the edit, hence, ignoring the original structure and content within the masked region. In this paper, we pursue an intuitive prompt-to-prompt editing framework, where the edits are controlled by text only. To this end, we analyze a text-conditioned model in depth and observe that the cross-attention layers are the key to controlling the relation between the spatial layout of the image to each word in the prompt. With this observation, we present several applications which monitor the image synthesis by editing the textual prompt only. This includes localized editing by replacing a word, global editing by adding a specification, and even delicately controlling the extent to which a word is reflected in the image. We present our results over diverse images and prompts, demonstrating high-quality synthesis and fidelity to the edited prompts. *

You can find additional information about Prompt-to-Prompt on the [project page](https://prompt-to-prompt.github.io/) and [original codebase](https://github.com/google/prompt-to-prompt/).
**TODO:** There is no demo yet. Add one?

<Tip>
Make sure to check out the Schedulers [guide](/using-diffusers/schedulers) to learn how to explore the tradeoff between scheduler speed and quality, and see the [reuse components across pipelines](/using-diffusers/loading#reuse-components-across-pipelines) section to learn how to efficiently load the same components into multiple pipelines.
</Tip>

## Usage example

### ReplaceEdit
```python
import torch
import numpy as np
import matplotlib.pyplot as plt
from diffusers.pipelines.prompt2prompt.pipeline_prompt2prompt import Prompt2PromptPipeline

pipe = Prompt2PromptPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to("cuda")

prompts = ["A turtle playing with a ball",
           "A monkey playing with a ball"]

edit_kwargs = {
    "cross_replace_steps": 0.4,
    "self_replace_steps": 0.4
}

outputs = pipe(prompt=prompts, height=512, width=512, num_inference_steps=NUM_50DIFFUSION_STEPS, edit_type='replace', edit_kwargs=edit_kwargs)
```

### ReplaceEdit with LocalBlend
```python
import torch
import numpy as np
import matplotlib.pyplot as plt
from diffusers.pipelines.prompt2prompt.pipeline_prompt2prompt import Prompt2PromptPipeline

pipe = Prompt2PromptPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to("cuda")

prompts = ["A turtle playing with a ball",
           "A monkey playing with a ball"]

edit_kwargs = {
    "cross_replace_steps": 0.4,
    "self_replace_steps": 0.4,
    "local_blend_words": ["turtle", "monkey"]
}

outputs = pipe(prompt=prompts, height=512, width=512, num_inference_steps=NUM_50DIFFUSION_STEPS, edit_type='replace', edit_kwargs=edit_kwargs)
```

### RefineEdit
```python
import torch
import numpy as np
import matplotlib.pyplot as plt
from diffusers.pipelines.prompt2prompt.pipeline_prompt2prompt import Prompt2PromptPipeline

pipe = Prompt2PromptPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to("cuda")

prompts = ["A turtle",
           "A turtle in a forest"]

edit_kwargs = {
    "cross_replace_steps": 0.4,
    "self_replace_steps": 0.4,
}

outputs = pipe(prompt=prompts, height=512, width=512, num_inference_steps=NUM_50DIFFUSION_STEPS, edit_type='refine', edit_kwargs=edit_kwargs)
```

### RefineEdit with LocalBlend
```python
import torch
import numpy as np
import matplotlib.pyplot as plt
from diffusers.pipelines.prompt2prompt.pipeline_prompt2prompt import Prompt2PromptPipeline

pipe = Prompt2PromptPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to("cuda")

prompts = ["A turtle",
           "A turtle in a forest"]

edit_kwargs = {
    "cross_replace_steps": 0.4,
    "self_replace_steps": 0.4,
    "local_blend_words": ["in", "a" , "forest"]
}

outputs = pipe(prompt=prompts, height=512, width=512, num_inference_steps=NUM_50DIFFUSION_STEPS, edit_type='refine', edit_kwargs=edit_kwargs)
```

### ReweightEdit
```python
import torch
import numpy as np
import matplotlib.pyplot as plt
from diffusers.pipelines.prompt2prompt.pipeline_prompt2prompt import Prompt2PromptPipeline

pipe = Prompt2PromptPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to("cuda")

prompts = ["A smiling turtle"] * 2

edit_kwargs = {
    "cross_replace_steps": 0.4,
    "self_replace_steps": 0.4,
    "equalizer_words": ["smiling"],
    "equalizer_strengths": [5]
}

outputs = pipe(prompt=prompts, height=512, width=512, num_inference_steps=NUM_50DIFFUSION_STEPS, edit_type='reweight', edit_kwargs=edit_kwargs)
```

## Prompt2PromptPipeline
[[autodoc]] Prompt2PromptPipeline
	- __call__

## StableDiffusionPipelineOutput
[[autodoc]] pipelines.stable_diffusion.StableDiffusionPipelineOutput