<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# 🧨 Diffusers의 윤리 지침 [[-diffusers-ethical-guidelines]]

## 서문 [[preamble]]

[Diffusers](https://huggingface.co/docs/diffusers/index)는 사전 훈련된 diffusion 모델을 제공하며, 추론과 훈련을 위한 모듈형 툴박스로 활용됩니다.

이 기술의 실제 적용 사례와 사회에 미칠 수 있는 잠재적 부정적 영향을 고려할 때, Diffusers 라이브러리의 개발, 사용자 기여, 사용에 윤리 지침을 제공하는 것이 중요하다고 생각합니다.

이 기술 사용과 관련된 위험은 여전히 검토 중이지만, 예를 들면: 예술가의 저작권 문제, 딥페이크 악용, 부적절한 맥락에서의 성적 콘텐츠 생성, 비동의 사칭, 소수자 집단 억압을 영속화하는 유해한 사회적 편견 등이 있습니다.
우리는 이러한 위험을 지속적으로 추적하고, 커뮤니티의 반응과 소중한 피드백에 따라 아래 지침을 조정할 것입니다.

## 범위 [[scope]]

Diffusers 커뮤니티는 프로젝트 개발에 다음 윤리 지침을 적용하며, 특히 윤리적 문제와 관련된 민감한 주제에 대해 커뮤니티의 기여를 조정하는 데 도움을 줄 것입니다.

## 윤리 지침 [[ethical-guidelines]]

다음 윤리 지침은 일반적으로 적용되지만, 윤리적으로 민감한 문제와 관련된 기술적 선택을 할 때 우선적으로 적용됩니다. 또한, 해당 기술의 최신 동향과 관련된 새로운 위험이 발생함에 따라 이러한 윤리 원칙을 지속적으로 조정할 것을 약속합니다.

- **투명성**: 우리는 PR 관리, 사용자에게 선택의 이유 설명, 기술적 의사결정 과정에서 투명성을 유지할 것을 약속합니다.

- **일관성**: 프로젝트 관리에서 모든 사용자에게 동일한 수준의 관심을 보장하고, 기술적으로 안정적이고 일관된 상태를 유지할 것을 약속합니다.

- **간결성**: Diffusers 라이브러리를 쉽게 사용하고 활용할 수 있도록, 프로젝트의 목표를 간결하고 일관성 있게 유지할 것을 약속합니다.

- **접근성**: Diffusers 프로젝트는 기술적 전문지식이 없어도 기여할 수 있도록 진입장벽을 낮춥니다. 이를 통해 연구 결과물이 커뮤니티에 더 잘 접근될 수 있습니다.

- **재현성**: 우리는 Diffusers 라이브러리를 통해 제공되는 업스트림 코드, 모델, 데이터셋의 재현성에 대해 투명하게 공개하는 것을 목표로 합니다.

- **책임**: 커뮤니티와 팀워크를 통해, 이 기술의 잠재적 위험을 예측하고 완화하는 데 공동 책임을 집니다.

## 구현 사례: 안전 기능과 메커니즘 [[examples-of-implementations-safety-features-and-mechanisms]]

팀은 diffusion 기술과 관련된 잠재적 윤리 및 사회적 위험에 대응하기 위해 기술적·비기술적 도구를 제공하고자 노력하고 있습니다. 또한, 커뮤니티의 참여는 이러한 기능 구현과 인식 제고에 매우 중요합니다.

- [**커뮤니티 탭**](https://huggingface.co/docs/hub/repositories-pull-requests-discussions): 커뮤니티가 프로젝트에 대해 토론하고 더 나은 협업을 할 수 있도록 지원합니다.

- **편향 탐색 및 평가**: Hugging Face 팀은 Stable Diffusion 모델의 편향성을 대화형으로 보여주는 [space](https://huggingface.co/spaces/society-ethics/DiffusionBiasExplorer)를 제공합니다. 우리는 이러한 편향 탐색과 평가를 지원하고 장려합니다.

- **배포에서의 안전 유도**

  - [**안전한 Stable Diffusion**](https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_diffusion/stable_diffusion_safe): 필터링되지 않은 웹 크롤링 데이터셋으로 훈련된 Stable Diffusion과 같은 모델이 부적절하게 변질되는 문제를 완화합니다. 관련 논문: [Safe Latent Diffusion: Mitigating Inappropriate Degeneration in Diffusion Models](https://huggingface.co/papers/2211.05105).

  - [**안전 검사기**](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/safety_checker.py): 생성된 이미지가 임베딩 공간에서 하드코딩된 유해 개념 클래스와 일치할 확률을 확인하고 비교합니다. 유해 개념은 역공학을 방지하기 위해 의도적으로 숨겨져 있습니다.

- **Hub에서의 단계적 배포**: 특히 민감한 상황에서는 일부 리포지토리에 대한 접근을 제한할 수 있습니다. 단계적 배포는 리포지토리 작성자가 사용에 대해 더 많은 통제권을 갖도록 하는 중간 단계입니다.

- **라이선싱**: [OpenRAILs](https://huggingface.co/blog/open_rail)와 같은 새로운 유형의 라이선스를 통해 자유로운 접근을 보장하면서도 보다 책임 있는 사용을 위한 일련의 제한을 둘 수 있습니다.
