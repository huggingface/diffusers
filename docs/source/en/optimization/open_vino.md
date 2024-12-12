<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# OpenVINO

🤗 [Optimum](https://github.com/huggingface/optimum-intel) provides Stable Diffusion pipelines compatible with OpenVINO to perform inference on a variety of Intel processors (see the [full list](https://docs.openvino.ai/latest/openvino_docs_OV_UG_supported_plugins_Supported_Devices.html) of supported devices).

You'll need to install 🤗 Optimum Intel with the `--upgrade-strategy eager` option to ensure [`optimum-intel`](https://github.com/huggingface/optimum-intel) is using the latest version:

```bash
pip install --upgrade-strategy eager optimum["openvino"]
```

This guide will show you how to use the Stable Diffusion and Stable Diffusion XL (SDXL) pipelines with OpenVINO.

## Stable Diffusion

To load and run inference, use the [`~optimum.intel.OVStableDiffusionPipeline`]. If you want to load a PyTorch model and convert it to the OpenVINO format on-the-fly, set `export=True`:

```python
from optimum.intel import OVStableDiffusionPipeline

model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
pipeline = OVStableDiffusionPipeline.from_pretrained(model_id, export=True)
prompt = "sailing ship in storm by Rembrandt"
image = pipeline(prompt).images[0]

# Don't forget to save the exported model
pipeline.save_pretrained("openvino-sd-v1-5")
```

To further speed-up inference, statically reshape the model. If you change any parameters such as the outputs height or width, you’ll need to statically reshape your model again.

```python
# Define the shapes related to the inputs and desired outputs
batch_size, num_images, height, width = 1, 1, 512, 512

# Statically reshape the model
pipeline.reshape(batch_size, height, width, num_images)
# Compile the model before inference
pipeline.compile()

image = pipeline(
    prompt,
    height=height,
    width=width,
    num_images_per_prompt=num_images,
).images[0]
```
<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/optimum/documentation-images/resolve/main/intel/openvino/stable_diffusion_v1_5_sail_boat_rembrandt.png">
</div>

You can find more examples in the 🤗 Optimum [documentation](https://huggingface.co/docs/optimum/intel/inference#stable-diffusion), and Stable Diffusion is supported for text-to-image, image-to-image, and inpainting.

## Stable Diffusion XL

To load and run inference with SDXL, use the [`~optimum.intel.OVStableDiffusionXLPipeline`]:

```python
from optimum.intel import OVStableDiffusionXLPipeline

model_id = "stabilityai/stable-diffusion-xl-base-1.0"
pipeline = OVStableDiffusionXLPipeline.from_pretrained(model_id)
prompt = "sailing ship in storm by Rembrandt"
image = pipeline(prompt).images[0]
```

To further speed-up inference, [statically reshape](#stable-diffusion) the model as shown in the Stable Diffusion section.

You can find more examples in the 🤗 Optimum [documentation](https://huggingface.co/docs/optimum/intel/inference#stable-diffusion-xl), and running SDXL in OpenVINO is supported for text-to-image and image-to-image.
