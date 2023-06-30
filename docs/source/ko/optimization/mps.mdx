<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Apple Silicon (M1/M2)에서 Stable Diffusion을 사용하는 방법

Diffusers는 Stable Diffusion 추론을 위해 PyTorch `mps`를 사용해 Apple 실리콘과 호환됩니다. 다음은 Stable Diffusion이 있는 M1 또는 M2 컴퓨터를 사용하기 위해 따라야 하는 단계입니다.

## 요구 사항

- Apple silicon (M1/M2) 하드웨어의 Mac 컴퓨터.
- macOS 12.6 또는 이후 (13.0 또는 이후 추천).
- Python arm64 버전
- PyTorch 2.0(추천) 또는 1.13(`mps`를 지원하는 최소 버전). Yhttps://pytorch.org/get-started/locally/의 지침에 따라 `pip` 또는 `conda`로 설치할 수 있습니다.


## 추론 파이프라인

아래 코도는 익숙한 `to()` 인터페이스를 사용하여 `mps` 백엔드로 Stable Diffusion 파이프라인을 M1 또는 M2 장치로 이동하는 방법을 보여줍니다.


<Tip warning={true}>

**PyTorch 1.13을 사용 중일 때 ** 추가 일회성 전달을 사용하여 파이프라인을 "프라이밍"하는 것을 추천합니다. 이것은 발견한 이상한 문제에 대한 임시 해결 방법입니다. 첫 번째 추론 전달은 후속 전달와 약간 다른 결과를 생성합니다. 이 전달은 한 번만 수행하면 되며 추론 단계를 한 번만 사용하고 결과를 폐기해도 됩니다.

</Tip>

이전 팁에서 설명한 것들을 포함한 여러 문제를 해결하므로 PyTorch 2 이상을 사용하는 것이 좋습니다.


```python
# `huggingface-cli login`에 로그인되어 있음을 확인
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe = pipe.to("mps")

# 컴퓨터가 64GB 이하의 RAM 램일 때 추천
pipe.enable_attention_slicing()

prompt = "a photo of an astronaut riding a horse on mars"

# 처음 "워밍업" 전달 (위 설명을 보세요)
_ = pipe(prompt, num_inference_steps=1)

# 결과는 워밍업 전달 후의 CPU 장치의 결과와 일치합니다.
image = pipe(prompt).images[0]
```

## 성능 추천

M1/M2 성능은 메모리 압력에 매우 민감합니다. 시스템은 필요한 경우 자동으로 스왑되지만 스왑할 때 성능이 크게 저하됩니다.


특히 컴퓨터의 시스템 RAM이 64GB 미만이거나 512 × 512픽셀보다 큰 비표준 해상도에서 이미지를 생성하는 경우, 추론 중에 메모리 압력을 줄이고 스와핑을 방지하기 위해 *어텐션 슬라이싱*을 사용하는 것이 좋습니다. 어텐션 슬라이싱은 비용이 많이 드는 어텐션 작업을 한 번에 모두 수행하는 대신 여러 단계로 수행합니다. 일반적으로 범용 메모리가 없는 컴퓨터에서 ~20%의 성능 영향을 미치지만 64GB 이상이 아닌 경우 대부분의 Apple Silicon 컴퓨터에서 *더 나은 성능*이 관찰되었습니다.

```python
pipeline.enable_attention_slicing()
```

## Known Issues

- 여러 프롬프트를 배치로 생성하는 것은 [충돌이 발생하거나 안정적으로 작동하지 않습니다](https://github.com/huggingface/diffusers/issues/363). 우리는 이것이 [PyTorch의 `mps` 백엔드](https://github.com/pytorch/pytorch/issues/84039)와 관련이 있다고 생각합니다. 이 문제는 해결되고 있지만 지금은 배치 대신 반복 방법을 사용하는 것이 좋습니다.