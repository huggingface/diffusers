<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# AutoModel

The `AutoModel` is designed to make it easy to load a checkpoint without needing to know the specific model class. `AutoModel` automatically retrieves the correct model class from the checkpoint `config.json` file.

```python
from diffusers import AutoModel, AutoPipelineForText2Image

unet = AutoModel.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="unet")
pipe = AutoPipelineForText2Image.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", unet=unet)
```


## AutoModel

[[autodoc]] AutoModel
	- all
	- from_pretrained
