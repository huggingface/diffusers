<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Text-guided depth-to-image 생성

[[open-in-colab]]

[`StableDiffusionDepth2ImgPipeline`]을 사용하면 텍스트 프롬프트와 초기 이미지를 전달하여 새 이미지의 생성을 조절할 수 있습니다. 또한 이미지 구조를 보존하기 위해 `depth_map`을 전달할 수도 있습니다. `depth_map`이 제공되지 않으면 파이프라인은 통합된 [depth-estimation model](https://github.com/isl-org/MiDaS)을 통해 자동으로 깊이를 예측합니다.


먼저 [`StableDiffusionDepth2ImgPipeline`]의 인스턴스를 생성합니다:

```python
import torch
import requests
from PIL import Image

from diffusers import StableDiffusionDepth2ImgPipeline

pipe = StableDiffusionDepth2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-depth",
    torch_dtype=torch.float16,
).to("cuda")
```

이제 프롬프트를 파이프라인에 전달합니다. 특정 단어가 이미지 생성을 가이드 하는것을 방지하기 위해 `negative_prompt`를 전달할 수도 있습니다:

```python
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
init_image = Image.open(requests.get(url, stream=True).raw)
prompt = "two tigers"
n_prompt = "bad, deformed, ugly, bad anatomy"
image = pipe(prompt=prompt, image=init_image, negative_prompt=n_prompt, strength=0.7).images[0]
image
```

| Input                                                                           | Output                                                                                                                                |
|---------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------|
| <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/coco-cats.png" width="500"/> | <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/depth2img-tigers.png" width="500"/> |

아래의 Spaces를 가지고 놀며 depth map이 있는 이미지와 없는 이미지의 차이가 있는지 확인해 보세요!

<iframe
	src="https://radames-stable-diffusion-depth2img.hf.space"
	frameborder="0"
	width="850"
	height="500"
></iframe>
