<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Text-guided 이미지 인페인팅(inpainting)

[[open-in-colab]]

[`StableDiffusionInpaintPipeline`]은 마스크와 텍스트 프롬프트를 제공하여 이미지의 특정 부분을 편집할 수 있도록 합니다. 이 기능은 인페인팅 작업을 위해 특별히 훈련된 [`runwayml/stable-diffusion-inpainting`](https://huggingface.co/runwayml/stable-diffusion-inpainting)과 같은 Stable Diffusion 버전을 사용합니다.

먼저 [`StableDiffusionInpaintPipeline`] 인스턴스를 불러옵니다:

```python
import PIL
import requests
import torch
from io import BytesIO

from diffusers import StableDiffusionInpaintPipeline

pipeline = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float16,
)
pipeline = pipeline.to("cuda")
```

나중에 교체할 강아지 이미지와 마스크를 다운로드하세요:

```python
def download_image(url):
    response = requests.get(url)
    return PIL.Image.open(BytesIO(response.content)).convert("RGB")


img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"

init_image = download_image(img_url).resize((512, 512))
mask_image = download_image(mask_url).resize((512, 512))
```

이제 마스크를 다른 것으로 교체하라는 프롬프트를 만들 수 있습니다:

```python
prompt = "Face of a yellow cat, high resolution, sitting on a park bench"
image = pipe(prompt=prompt, image=init_image, mask_image=mask_image).images[0]
```

`image`          | `mask_image` | `prompt` | output |
:-------------------------:|:-------------------------:|:-------------------------:|-------------------------:|
<img src="https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png" alt="drawing" width="250"/> | <img src="https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png" alt="drawing" width="250"/> | ***Face of a yellow cat, high resolution, sitting on a park bench*** | <img src="https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/in_paint/yellow_cat_sitting_on_a_park_bench.png" alt="drawing" width="250"/> |

<Tip warning={true}>

이전의 실험적인 인페인팅 구현에서는 품질이 낮은 다른 프로세스를 사용했습니다. 이전 버전과의 호환성을 보장하기 위해 새 모델이 포함되지 않은 사전학습된 파이프라인을 불러오면 이전 인페인팅 방법이 계속 적용됩니다.

</Tip>

아래 Space에서 이미지 인페인팅을 직접 해보세요!

<iframe
  src="https://runwayml-stable-diffusion-inpainting.hf.space"
  frameborder="0"
  width="850"
  height="500"
></iframe>
