<!--Copyright 2024 The HuggingFace Team, The Black Forest Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# FluxControlInpaint

FluxControlInpaintPipeline is an implementation of Inpainting for Flux.1 Depth/Canny models. It is a pipeline that allows you to inpaint images using the Flux.1 Depth/Canny models. The pipeline takes an image and a mask as input and returns the inpainted image.

FLUX.1 Depth and Canny [dev] is a 12 billion parameter rectified flow transformer capable of generating an image based on a text description while following the structure of a given input image. **This is not a ControlNet model**.

| Control type | Developer | Link |
| -------- | ---------- | ---- |
| Depth | [Black Forest Labs](https://huggingface.co/black-forest-labs) | [Link](https://huggingface.co/black-forest-labs/FLUX.1-Depth-dev) |
| Canny | [Black Forest Labs](https://huggingface.co/black-forest-labs) | [Link](https://huggingface.co/black-forest-labs/FLUX.1-Canny-dev) |


<Tip>

Flux can be quite expensive to run on consumer hardware devices. However, you can perform a suite of optimizations to run it faster and in a more memory-friendly manner. Check out [this section](https://huggingface.co/blog/sd3#memory-optimizations-for-sd3) for more details. Additionally, Flux can benefit from quantization for memory efficiency with a trade-off in inference latency. Refer to [this blog post](https://huggingface.co/blog/quanto-diffusers) to learn more. For an exhaustive list of resources, check out [this gist](https://gist.github.com/sayakpaul/b664605caf0aa3bf8585ab109dd5ac9c).

</Tip>

```python
import torch
from diffusers import FluxControlInpaintPipeline
from diffusers.models.transformers import FluxTransformer2DModel
from transformers import T5EncoderModel
from diffusers.utils import load_image, make_image_grid
from image_gen_aux import DepthPreprocessor # https://github.com/huggingface/image_gen_aux
from PIL import Image
import numpy as np

pipe = FluxControlInpaintPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Depth-dev",
    torch_dtype=torch.bfloat16,
)
# use following lines if you have GPU constraints
# ---------------------------------------------------------------
transformer = FluxTransformer2DModel.from_pretrained(
    "sayakpaul/FLUX.1-Depth-dev-nf4", subfolder="transformer", torch_dtype=torch.bfloat16
)
text_encoder_2 = T5EncoderModel.from_pretrained(
    "sayakpaul/FLUX.1-Depth-dev-nf4", subfolder="text_encoder_2", torch_dtype=torch.bfloat16
)
pipe.transformer = transformer
pipe.text_encoder_2 = text_encoder_2
pipe.enable_model_cpu_offload()
# ---------------------------------------------------------------
pipe.to("cuda")

prompt = "a blue robot singing opera with human-like expressions"
image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/robot.png")

head_mask = np.zeros_like(image)
head_mask[65:580,300:642] = 255
mask_image = Image.fromarray(head_mask)

processor = DepthPreprocessor.from_pretrained("LiheYoung/depth-anything-large-hf")
control_image = processor(image)[0].convert("RGB")

output = pipe(
    prompt=prompt,
    image=image,
    control_image=control_image,
    mask_image=mask_image,
    num_inference_steps=30,
    strength=0.9,
    guidance_scale=10.0,
    generator=torch.Generator().manual_seed(42),
).images[0]
make_image_grid([image, control_image, mask_image, output.resize(image.size)], rows=1, cols=4).save("output.png")
```

## FluxControlInpaintPipeline
[[autodoc]] FluxControlInpaintPipeline
	- all
	- __call__


## FluxPipelineOutput
[[autodoc]] pipelines.flux.pipeline_output.FluxPipelineOutput