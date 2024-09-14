<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# 새로운 작업에 대한 모델을 적용하기

많은 diffusion 시스템은 같은 구성 요소들을 공유하므로 한 작업에 대해 사전학습된 모델을 완전히 다른 작업에 적용할 수 있습니다.

이 인페인팅을 위한 가이드는 사전학습된 [`UNet2DConditionModel`]의 아키텍처를 초기화하고 수정하여 사전학습된 text-to-image 모델을 어떻게 인페인팅에 적용하는지를 알려줄 것입니다.

## UNet2DConditionModel 파라미터 구성

[`UNet2DConditionModel`]은 [input sample](https://huggingface.co/docs/diffusers/v0.16.0/en/api/models#diffusers.UNet2DConditionModel.in_channels)에서 4개의 채널을 기본적으로 허용합니다. 예를 들어,  [`stable-diffusion-v1-5/stable-diffusion-v1-5`](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)와 같은 사전학습된 text-to-image 모델을 불러오고 `in_channels`의 수를 확인합니다:

```py
from diffusers import StableDiffusionPipeline

pipeline = StableDiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5")
pipeline.unet.config["in_channels"]
4
```

인페인팅은 입력 샘플에 9개의 채널이 필요합니다. [`runwayml/stable-diffusion-inpainting`](https://huggingface.co/runwayml/stable-diffusion-inpainting)와 같은 사전학습된 인페인팅 모델에서 이 값을 확인할 수 있습니다:

```py
from diffusers import StableDiffusionPipeline

pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-inpainting")
pipeline.unet.config["in_channels"]
9
```

인페인팅에 대한 text-to-image 모델을 적용하기 위해, `in_channels` 수를 4에서 9로 수정해야 할 것입니다.

사전학습된 text-to-image 모델의 가중치와 [`UNet2DConditionModel`]을 초기화하고 `in_channels`를 9로 수정해 주세요. `in_channels`의 수를 수정하면 크기가 달라지기 때문에 크기가 안 맞는 오류를 피하기 위해 `ignore_mismatched_sizes=True` 및 `low_cpu_mem_usage=False`를 설정해야 합니다.

```py
from diffusers import UNet2DConditionModel

model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
unet = UNet2DConditionModel.from_pretrained(
    model_id, subfolder="unet", in_channels=9, low_cpu_mem_usage=False, ignore_mismatched_sizes=True
)
```

Text-to-image 모델로부터 다른 구성 요소의 사전학습된 가중치는 체크포인트로부터 초기화되지만 `unet`의 입력 채널 가중치 (`conv_in.weight`)는 랜덤하게 초기화됩니다. 그렇지 않으면 모델이 노이즈를 리턴하기 때문에 인페인팅의 모델을 파인튜닝 할 때 중요합니다.
