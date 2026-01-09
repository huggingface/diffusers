<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# xFormers 설치하기

추론과 학습 모두에 [xFormers](https://github.com/facebookresearch/xformers)를 사용하는 것이 좋습니다.
자체 테스트로 어텐션 블록에서 수행된 최적화가 더 빠른 속도와 적은 메모리 소비를 확인했습니다.

2023년 1월에 출시된 xFormers 버전 '0.0.16'부터 사전 빌드된 pip wheel을 사용하여 쉽게 설치할 수 있습니다:

```bash
pip install xformers
```

> [!TIP]
> xFormers PIP 패키지에는 최신 버전의 PyTorch(xFormers 0.0.16에 1.13.1)가 필요합니다. 이전 버전의 PyTorch를 사용해야 하는 경우 [프로젝트 지침](https://github.com/facebookresearch/xformers#installing-xformers)의 소스를 사용해 xFormers를 설치하는 것이 좋습니다.

xFormers를 설치하면, [여기](fp16#memory-efficient-attention)서 설명한 것처럼 'enable_xformers_memory_efficient_attention()'을 사용하여 추론 속도를 높이고 메모리 소비를 줄일 수 있습니다.

> [!WARNING]
> [이 이슈](https://github.com/huggingface/diffusers/issues/2234#issuecomment-1416931212)에 따르면 xFormers `v0.0.16`에서 GPU를 사용한 학습(파인 튜닝 또는 Dreambooth)을 할 수 없습니다. 해당 문제가 발견되면. 해당 코멘트를 참고해 development 버전을 설치하세요.
