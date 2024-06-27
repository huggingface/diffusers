<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# 🧨 Diffusers의 윤리 지침

## 서문

[Diffusers](https://huggingface.co/docs/diffusers/index)는 사전 훈련된 diffusion 모델을 제공하며 추론 및 훈련을 위한 모듈식 툴박스로 사용됩니다.

이 기술의 실제 적용과 사회에 미칠 수 있는 부정적인 영향을 고려하여 Diffusers 라이브러리의 개발, 사용자 기여 및 사용에 윤리 지침을 제공하는 것이 중요하다고 생각합니다.

이 기술을 사용하는 데 연관된 위험은 아직 조사 중이지만, 몇 가지 예를 들면: 예술가들에 대한 저작권 문제; 딥 페이크의 악용; 부적절한 맥락에서의 성적 콘텐츠 생성; 동의 없는 impersonation; 사회적인 편견으로 인해 억압되는 그룹들에 대한 해로운 영향입니다.
우리는 위험을 지속적으로 추적하고 커뮤니티의 응답과 소중한 피드백에 따라 다음 지침을 조정할 것입니다.


## 범위

Diffusers 커뮤니티는 프로젝트의 개발에 다음과 같은 윤리 지침을 적용하며, 특히 윤리적 문제와 관련된 민감한 주제에 대한 커뮤니티의 기여를 조정하는 데 도움을 줄 것입니다.


## 윤리 지침

다음 윤리 지침은 일반적으로 적용되지만, 기술적 선택을 할 때 윤리적으로 민감한 문제를 다룰 때 주로 적용할 것입니다. 또한, 해당 기술의 최신 동향과 관련된 신규 위험에 따라 시간이 지남에 따라 이러한 윤리 원칙을 조정할 것을 약속합니다.

- **투명성**: 우리는 PR을 관리하고, 사용자에게 우리의 선택을 설명하며, 기술적 의사결정을 내릴 때 투명성을 유지할 것을 약속합니다.

- **일관성**: 우리는 프로젝트 관리에서 사용자들에게 동일한 수준의 관심을 보장하고 기술적으로 안정되고 일관된 상태를 유지할 것을 약속합니다.

- **간결성**: Diffusers 라이브러리를 사용하고 활용하기 쉽게 만들기 위해, 프로젝트의 목표를 간결하고 일관성 있게 유지할 것을 약속합니다.

- **접근성**: Diffusers 프로젝트는 기술적 전문 지식 없어도 프로젝트 운영에 참여할 수 있는 기여자의 진입장벽을 낮춥니다. 이를 통해 연구 결과물이 커뮤니티에 더 잘 접근할 수 있게 됩니다.

- **재현성**: 우리는 Diffusers 라이브러리를 통해 제공되는 업스트림(upstream) 코드, 모델 및 데이터셋의 재현성에 대해 투명하게 공개할 것을 목표로 합니다.

- **책임**: 우리는 커뮤니티와 팀워크를 통해, 이 기술의 잠재적인 위험과 위험을 예측하고 완화하는 데 대한 공동 책임을 가지고 있습니다.


## 구현 사례: 안전 기능과 메커니즘

팀은 diffusion 기술과 관련된 잠재적인 윤리 및 사회적 위험에 대처하기 위한 기술적 및 비기술적 도구를 제공하고자 하고 있습니다. 또한, 커뮤니티의 참여는 이러한 기능의 구현하고 우리와 함께 인식을 높이는 데 매우 중요합니다.

- [**커뮤니티 탭**](https://huggingface.co/docs/hub/repositories-pull-requests-discussions): 이를 통해 커뮤니티는 프로젝트에 대해 토론하고 더 나은 협력을 할 수 있습니다.

- **편향 탐색 및 평가**: Hugging Face 팀은 Stable Diffusion 모델의 편향성을 대화형으로 보여주는 [space](https://huggingface.co/spaces/society-ethics/DiffusionBiasExplorer)을 제공합니다. 이런 의미에서, 우리는 편향 탐색 및 평가를 지원하고 장려합니다.

- **배포에서의 안전 유도**

  - [**안전한 Stable Diffusion**](https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_diffusion/stable_diffusion_safe): 이는 필터되지 않은 웹 크롤링 데이터셋으로 훈련된 Stable Diffusion과 같은 모델이 부적절한 변질에 취약한 문제를 완화합니다. 관련 논문: [Safe Latent Diffusion: Mitigating Inappropriate Degeneration in Diffusion Models](https://arxiv.org/abs/2211.05105).

  - [**안전 검사기**](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/safety_checker.py): 이미지가 생성된 후에 이미자가 임베딩 공간에서 일련의 하드코딩된 유해 개념의 클래스일 확률을 확인하고 비교합니다. 유해 개념은 역공학을 방지하기 위해 의도적으로 숨겨져 있습니다.

- **Hub에서의 단계적인 배포**: 특히 민감한 상황에서는 일부 리포지토리에 대한 접근을 제한해야 합니다. 이 단계적인 배포는 중간 단계로, 리포지토리 작성자가 사용에 대한 더 많은 통제력을 갖게 합니다.

- **라이선싱**: [OpenRAILs](https://huggingface.co/blog/open_rail)와 같은 새로운 유형의 라이선싱을 통해 자유로운 접근을 보장하면서도 더 책임 있는 사용을 위한 일련의 제한을 둘 수 있습니다.
