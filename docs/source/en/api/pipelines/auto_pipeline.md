<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# AutoPipeline

In many cases, one checkpoint can be used for multiple tasks. For example, you may be able to use the same checkpoint for Text-to-Image, Image-to-Image, and Inpainting. However, you'll need to know the pipeline class names linked to your checkpoint. 

AutoPipeline is designed to make it easy for you to use multiple pipelines in your workflow. We currently provide 3 AutoPipeline classes to perform three different tasks, i.e. [`AutoPipelineForText2Image`], [`AutoPipelineForImage2Image`], and [`AutoPipelineForInpainting`]. You'll need to choose the AutoPipeline class based on the task you want to perform and use it to automatically retrieve the relevant pipeline given the name/path to the pre-trained weights. 

For example, to perform Image-to-Image with the SD1.5 checkpoint, you can do

```python
from diffusers import PipelineForImageToImage

pipe_i2i = PipelineForImageoImage.from_pretrained("runwayml/stable-diffusion-v1-5")
```

It will also help you switch between tasks seamlessly using the same checkpoint without reallocating additional memory. For example, to re-use the Image-to-Image pipeline we just created for inpainting, you can do 

```python
from diffusers import PipelineForInpainting

pipe_inpaint = AutoPipelineForInpainting.from_pipe(pipe_i2i)
```
All the components will be transferred to the inpainting pipeline with zero cost.


Currently AutoPipeline support the Text-to-Image, Image-to-Image, and Inpainting tasks for below diffusion models:
- [stable Diffusion](./stable_diffusion)
- [Stable Diffusion Controlnet](./api/pipelines/controlnet)
- [Stable Diffusion XL](./stable_diffusion/stable_diffusion_xl)
- [IF](./if) 
- [Kandinsky](./kandinsky)
- [Kandinsky 2.2](./kandinsky)


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


