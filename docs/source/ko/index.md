<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

<p align="center">
    <br>
    <img src="https://raw.githubusercontent.com/huggingface/diffusers/77aadfee6a891ab9fcfb780f87c693f7a5beeb8e/docs/source/imgs/diffusers_library.jpg" width="400"/>
    <br>
</p>


# Diffusers

🤗 Diffusers는 이미지, 오디오, 심지어 분자의 3D 구조를 생성하기 위한 최첨단 사전 훈련된 diffusion 모델을 위한 라이브러리입니다. 간단한 추론 솔루션을 찾고 있든, 자체 diffusion 모델을 훈련하고 싶든, 🤗 Diffusers는 두 가지 모두를 지원하는 모듈식 툴박스입니다. 저희 라이브러리는 [성능보다 사용성](conceptual/philosophy#usability-over-performance), [간편함보다 단순함](conceptual/philosophy#simple-over-easy), 그리고 [추상화보다 사용자 지정 가능성](conceptual/philosophy#tweakable-contributorfriendly-over-abstraction)에 중점을 두고 설계되었습니다.

이 라이브러리에는 세 가지 주요 구성 요소가 있습니다:

- 몇 줄의 코드만으로 추론할 수 있는 최첨단 [diffusion 파이프라인](api/pipelines/overview).
- 생성 속도와 품질 간의 균형을 맞추기 위해 상호교환적으로 사용할 수 있는 [노이즈 스케줄러](api/schedulers/overview).
- 빌딩 블록으로 사용할 수 있고 스케줄러와 결합하여 자체적인 end-to-end diffusion 시스템을 만들 수 있는 사전 학습된 [모델](api/models).

<div class="mt-10">
  <div class="w-full flex flex-col space-y-4 md:space-y-0 md:grid md:grid-cols-2 md:gap-y-4 md:gap-x-5">
    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg" href="./tutorials/tutorial_overview"
      ><div class="w-full text-center bg-gradient-to-br from-blue-400 to-blue-500 rounded-lg py-1.5 font-semibold mb-5 text-white text-lg leading-relaxed">Tutorials</div>
      <p class="text-gray-700">결과물을 생성하고, 나만의 diffusion 시스템을 구축하고, 확산 모델을 훈련하는 데 필요한 기본 기술을 배워보세요. 🤗 Diffusers를 처음 사용하는 경우 여기에서 시작하는 것이 좋습니다!</p>
    </a>
    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg" href="./using-diffusers/loading_overview"
      ><div class="w-full text-center bg-gradient-to-br from-indigo-400 to-indigo-500 rounded-lg py-1.5 font-semibold mb-5 text-white text-lg leading-relaxed">How-to guides</div>
      <p class="text-gray-700">파이프라인, 모델, 스케줄러를 로드하는 데 도움이 되는 실용적인 가이드입니다. 또한 특정 작업에 파이프라인을 사용하고, 출력 생성 방식을 제어하고, 추론 속도에 맞게 최적화하고, 다양한 학습 기법을 사용하는 방법도 배울 수 있습니다.</p>
    </a>
    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg" href="./conceptual/philosophy"
      ><div class="w-full text-center bg-gradient-to-br from-pink-400 to-pink-500 rounded-lg py-1.5 font-semibold mb-5 text-white text-lg leading-relaxed">Conceptual guides</div>
      <p class="text-gray-700">라이브러리가 왜 이런 방식으로 설계되었는지 이해하고, 라이브러리 이용에 대한 윤리적 가이드라인과 안전 구현에 대해 자세히 알아보세요.</p>
   </a>
    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg" href="./api/models"
      ><div class="w-full text-center bg-gradient-to-br from-purple-400 to-purple-500 rounded-lg py-1.5 font-semibold mb-5 text-white text-lg leading-relaxed">Reference</div>
      <p class="text-gray-700">🤗 Diffusers 클래스 및 메서드의 작동 방식에 대한 기술 설명.</p>
    </a>
  </div>
</div>