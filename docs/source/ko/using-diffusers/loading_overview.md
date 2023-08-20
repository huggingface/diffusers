<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Overview

🧨 Diffusers는 생성 작업을 위한 다양한 파이프라인, 모델, 스케줄러를 제공합니다. 이러한 컴포넌트를 최대한 간단하게 로드할 수 있도록 단일 통합 메서드인 `from_pretrained()`를 제공하여 Hugging Face [Hub](https://huggingface.co/models?library=diffusers&sort=downloads) 또는 로컬 머신에서 이러한 컴포넌트를 불러올 수 있습니다. 파이프라인이나 모델을 로드할 때마다, 최신 파일이 자동으로 다운로드되고 캐시되므로, 다음에 파일을 다시 다운로드하지 않고도 빠르게 재사용할 수 있습니다.

이 섹션은 파이프라인 로딩, 파이프라인에서 다양한 컴포넌트를 로드하는 방법, 체크포인트 variants를 불러오는 방법, 그리고 커뮤니티 파이프라인을 불러오는 방법에 대해 알아야 할 모든 것들을 다룹니다. 또한 스케줄러를 불러오는 방법과 서로 다른 스케줄러를 사용할 때 발생하는 속도와 품질간의 트레이드 오프를 비교하는 방법 역시 다룹니다. 그리고 마지막으로 🧨 Diffusers와 함께 파이토치에서 사용할 수 있도록 KerasCV 체크포인트를 변환하고 불러오는 방법을 살펴봅니다.

