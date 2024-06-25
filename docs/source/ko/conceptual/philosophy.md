<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# 철학 [[philosophy]] 

🧨 Diffusers는 다양한 모달리티에서 **최신의** 사전 훈련된 diffusion 모델을 제공합니다.
그 목적은 추론과 훈련을 위한 **모듈식 툴박스**로 사용되는 것입니다.

저희는 시간이 지나도 변치 않는 라이브러리를 구축하는 것을 목표로 하기에 API 설계를 매우 중요하게 생각합니다.

간단히 말해서, Diffusers는 PyTorch를 자연스럽게 확장할 수 있도록 만들어졌습니다. 따라서 대부분의 설계 선택은 [PyTorch의 설계 원칙](https://pytorch.org/docs/stable/community/design.html#pytorch-design-philosophy)에 기반합니다. 이제 가장 중요한 것들을 살펴보겠습니다:

## 성능보다는 사용성을 [[usability-over-performance]]

- Diffusers는 다양한 성능 향상 기능이 내장되어 있지만 (자세한 내용은 [메모리와 속도](https://huggingface.co/docs/diffusers/optimization/fp16) 참조), 모델은 항상 가장 높은 정밀도와 최소한의 최적화로 로드됩니다. 따라서 사용자가 별도로 정의하지 않는 한 기본적으로 diffusion 파이프라인은 항상 float32 정밀도로 CPU에 인스턴스화됩니다. 이는 다양한 플랫폼과 가속기에서의 사용성을 보장하며, 라이브러리를 실행하기 위해 복잡한 설치가 필요하지 않다는 것을 의미합니다.
- Diffusers는 **가벼운** 패키지를 지향하기 때문에 필수 종속성은 거의 없지만 성능을 향상시킬 수 있는 많은 선택적 종속성이 있습니다 (`accelerate`, `safetensors`, `onnx` 등). 저희는 라이브러리를 가능한 한 가볍게 유지하여 다른 패키지에 대한 종속성 걱정이 없도록 노력하고 있습니다.
- Diffusers는 간결하고 이해하기 쉬운 코드를 선호합니다. 이는 람다 함수나 고급 PyTorch 연산자와 같은 압축된 코드 구문을 자주 사용하지 않는 것을 의미합니다.

## 쉬움보다는 간단함을 [[simple-over-easy]]

PyTorch에서는 **명시적인 것이 암시적인 것보다 낫다**와 **단순한 것이 복잡한 것보다 낫다**라고 말합니다. 이 설계 철학은 라이브러리의 여러 부분에 반영되어 있습니다:
- [`DiffusionPipeline.to`](https://huggingface.co/docs/diffusers/main/en/api/diffusion_pipeline#diffusers.DiffusionPipeline.to)와 같은 메소드를 사용하여 사용자가 장치 관리를 할 수 있도록 PyTorch의 API를 따릅니다.
- 잘못된 입력을 조용히 수정하는 대신 간결한 오류 메시지를 발생시키는 것이 우선입니다. Diffusers는 라이브러리를 가능한 한 쉽게 사용할 수 있도록 하는 것보다 사용자를 가르치는 것을 목표로 합니다.
- 복잡한 모델과 스케줄러 로직이 내부에서 마법처럼 처리하는 대신 노출됩니다. 스케줄러/샘플러는 서로에게 최소한의 종속성을 가지고 분리되어 있습니다. 이로써 사용자는 언롤된 노이즈 제거 루프를 작성해야 합니다. 그러나 이 분리는 디버깅을 더 쉽게하고 노이즈 제거 과정을 조정하거나 diffusers 모델이나 스케줄러를 교체하는 데 사용자에게 더 많은 제어권을 제공합니다.
- diffusers 파이프라인의 따로 훈련된 구성 요소인 text encoder, unet 및 variational autoencoder는 각각 자체 모델 클래스를 갖습니다. 이로써 사용자는 서로 다른 모델의 구성 요소 간의 상호 작용을 처리해야 하며, 직렬화 형식은 모델 구성 요소를 다른 파일로 분리합니다. 그러나 이는 디버깅과 커스터마이징을 더 쉽게합니다. DreamBooth나 Textual Inversion 훈련은 Diffusers의 'diffusion 파이프라인의 단일 구성 요소들을 분리할 수 있는 능력' 덕분에 매우 간단합니다.

## 추상화보다는 수정 가능하고 기여하기 쉬움을 [[tweakable-contributor-friendly-over-abstraction]]

라이브러리의 대부분에 대해 Diffusers는 [Transformers 라이브러리](https://github.com/huggingface/transformers)의 중요한 설계 원칙을 채택합니다, 바로 성급한 추상화보다는 copy-pasted 코드를 선호한다는 것입니다. 이 설계 원칙은 [Don't repeat yourself (DRY)](https://en.wikipedia.org/wiki/Don%27t_repeat_yourself)와 같은 인기 있는 설계 원칙과는 대조적으로 매우 의견이 분분한데요.
간단히 말해서, Transformers가 모델링 파일에 대해 수행하는 것처럼, Diffusers는 매우 낮은 수준의 추상화와 매우 독립적인 코드를 유지하는 것을 선호합니다. 함수, 긴 코드 블록, 심지어 클래스도 여러 파일에 복사할 수 있으며, 이는 처음에는 라이브러리를 유지할 수 없게 만드는 나쁜, 서투른 설계 선택으로 보일 수 있습니다. 하지만 이러한 설계는 매우 성공적이며, 커뮤니티 기반의 오픈 소스 기계 학습 라이브러리에 매우 적합합니다. 그 이유는 다음과 같습니다:
- 기계 학습은 패러다임, 모델 아키텍처 및 알고리즘이 빠르게 변화하는 매우 빠르게 움직이는 분야이기 때문에 오랜 기간 지속되는 코드 추상화를 정의하기가 매우 어렵습니다.
- 기계 학습 전문가들은 아이디어와 연구를 위해 기존 코드를 빠르게 조정할 수 있어야 하므로, 많은 추상화보다는 독립적인 코드를 선호합니다.
- 오픈 소스 라이브러리는 커뮤니티 기여에 의존하므로, 기여하기 쉬운 라이브러리를 구축해야 합니다. 코드가 추상화되면 의존성이 많아지고 읽기 어렵고 기여하기 어려워집니다. 기여자들은 중요한 기능을 망가뜨릴까 두려워하여 매우 추상화된 라이브러리에 기여하지 않게 됩니다. 라이브러리에 기여하는 것이 다른 기본 코드를 망가뜨릴 수 없다면, 잠재적인 새로운 기여자에게 더욱 환영받을 수 있을 뿐만 아니라 여러 부분에 대해 병렬적으로 검토하고 기여하기가 더 쉬워집니다.

Hugging Face에서는 이 설계를 **단일 파일 정책**이라고 부르며, 특정 클래스의 대부분의 코드가 단일하고 독립적인 파일에 작성되어야 한다는 의미입니다. 철학에 대해 자세히 알아보려면 [이 블로그 글](https://huggingface.co/blog/transformers-design-philosophy)을 참조할 수 있습니다.

Diffusers에서는 이러한 철학을 파이프라인과 스케줄러에 모두 따르지만, diffusion 모델에 대해서는 일부만 따릅니다. 일부만 따르는 이유는 Diffusion 파이프라인인 [DDPM](https://huggingface.co/docs/diffusers/api/pipelines/ddpm), [Stable Diffusion](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/overview#stable-diffusion-pipelines), [unCLIP (DALL·E 2)](https://huggingface.co/docs/diffusers/api/pipelines/unclip) 및 [Imagen](https://imagen.research.google/) 등 대부분의 diffusion 파이프라인은 동일한 diffusion 모델인 [UNet](https://huggingface.co/docs/diffusers/api/models/unet2d-cond)에 의존하기 때문입니다.

좋아요, 이제 🧨 Diffusers가 설계된 방식을 대략적으로 이해했을 것입니다 🤗.
우리는 이러한 설계 원칙을 일관되게 라이브러리 전체에 적용하려고 노력하고 있습니다. 그럼에도 불구하고 철학에 대한 일부 예외 사항이나 불행한 설계 선택이 있을 수 있습니다. 디자인에 대한 피드백이 있다면 [GitHub에서 직접](https://github.com/huggingface/diffusers/issues/new?assignees=&labels=&template=feedback.md&title=) 알려주시면 감사하겠습니다.

## 디자인 철학 자세히 알아보기 [[design-philosophy-in-details]]

이제 디자인 철학의 세부 사항을 좀 더 자세히 살펴보겠습니다. Diffusers는 주로 세 가지 주요 클래스로 구성됩니다: [파이프라인](https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines), [모델](https://github.com/huggingface/diffusers/tree/main/src/diffusers/models), 그리고 [스케줄러](https://github.com/huggingface/diffusers/tree/main/src/diffusers/schedulers). 각 클래스에 대한 더 자세한 설계 결정 사항을 살펴보겠습니다.

### 파이프라인 [[pipelines]]

파이프라인은 사용하기 쉽도록 설계되었으며 (따라서 [*쉬움보다는 간단함을*](#쉬움보다는-간단함을)을 100% 따르지는 않음), feature-complete하지 않으며, 추론을 위한 [모델](#모델)과 [스케줄러](#스케줄러)를 사용하는 방법의 예시로 간주될 수 있습니다.

다음과 같은 설계 원칙을 따릅니다:
- 파이프라인은 단일 파일 정책을 따릅니다. 모든 파이프라인은 src/diffusers/pipelines의 개별 디렉토리에 있습니다. 하나의 파이프라인 폴더는 하나의 diffusion 논문/프로젝트/릴리스에 해당합니다. 여러 파이프라인 파일은 하나의 파이프라인 폴더에 모을 수 있습니다. 예를 들어 [`src/diffusers/pipelines/stable-diffusion`](https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines/stable_diffusion)에서 그렇게 하고 있습니다. 파이프라인이 유사한 기능을 공유하는 경우, [# Copied from mechanism](https://github.com/huggingface/diffusers/blob/125d783076e5bd9785beb05367a2d2566843a271/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_img2img.py#L251)을 사용할 수 있습니다.
- 파이프라인은 모두 [`DiffusionPipeline`]을 상속합니다.
- 각 파이프라인은 서로 다른 모델 및 스케줄러 구성 요소로 구성되어 있으며, 이는 [`model_index.json` 파일](https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/model_index.json)에 문서화되어 있으며, 파이프라인의 속성 이름과 동일한 이름으로 액세스할 수 있으며, [`DiffusionPipeline.components`](https://huggingface.co/docs/diffusers/main/en/api/diffusion_pipeline#diffusers.DiffusionPipeline.components) 함수를 통해 파이프라인 간에 공유할 수 있습니다.
- 각 파이프라인은 [`DiffusionPipeline.from_pretrained`](https://huggingface.co/docs/diffusers/main/en/api/diffusion_pipeline#diffusers.DiffusionPipeline.from_pretrained) 함수를 통해 로드할 수 있어야 합니다.
- 파이프라인은 추론에**만** 사용되어야 합니다.
- 파이프라인은 매우 가독성이 좋고, 이해하기 쉽고, 쉽게 조정할 수 있도록 설계되어야 합니다.
- 파이프라인은 서로 상호작용하고, 상위 수준 API에 쉽게 통합할 수 있도록 설계되어야 합니다.
- 파이프라인은 사용자 인터페이스가 feature-complete하지 않게 하는 것을 목표로 합니다. future-complete한 사용자 인터페이스를 원한다면 [InvokeAI](https://github.com/invoke-ai/InvokeAI), [Diffuzers](https://github.com/abhishekkrthakur/diffuzers), [lama-cleaner](https://github.com/Sanster/lama-cleaner)를 참조해야 합니다.
- 모든 파이프라인은 오로지 `__call__` 메소드를 통해 실행할 수 있어야 합니다. `__call__` 인자의 이름은 모든 파이프라인에서 공유되어야 합니다.
- 파이프라인은 해결하고자 하는 작업의 이름으로 지정되어야 합니다.
- 대부분의 경우에 새로운 diffusion 파이프라인은 새로운 파이프라인 폴더/파일에 구현되어야 합니다.

### 모델 [[models]]

모델은 [PyTorch의 Module 클래스](https://pytorch.org/docs/stable/generated/torch.nn.Module.html)의 자연스러운 확장이 되도록, 구성 가능한 툴박스로 설계되었습니다. 그리고 모델은 **단일 파일 정책**을 일부만 따릅니다.

다음과 같은 설계 원칙을 따릅니다:
- 모델은 **모델 아키텍처 유형**에 해당합니다. 예를 들어 [`UNet2DConditionModel`] 클래스는 2D 이미지 입력을 기대하고 일부 context에 의존하는 모든 UNet 변형들에 사용됩니다.
- 모든 모델은 [`src/diffusers/models`](https://github.com/huggingface/diffusers/tree/main/src/diffusers/models)에서 찾을 수 있으며, 각 모델 아키텍처는 해당 파일에 정의되어야 합니다. 예를 들어 [`unet_2d_condition.py`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/unet_2d_condition.py), [`transformer_2d.py`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformer_2d.py) 등이 있습니다.
- 모델은 **단일 파일 정책**을 따르지 않으며, [`attention.py`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention.py), [`resnet.py`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/resnet.py), [`embeddings.py`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/embeddings.py) 등과 같은 작은 모델 구성 요소를 사용해야 합니다. **참고**: 이는 Transformers의 모델링 파일과는 대조적으로 모델이 실제로 단일 파일 정책을 따르지 않음을 보여줍니다.
- 모델은 PyTorch의 `Module` 클래스와 마찬가지로 복잡성을 노출하고 명확한 오류 메시지를 제공해야 합니다.
- 모든 모델은 `ModelMixin`과 `ConfigMixin`을 상속합니다.
- 모델은 주요 코드 변경이 필요하지 않고, 역호환성을 유지하며, 메모리 또는 컴퓨팅과 관련한 중요한 이득을 제공할 때 성능을 위해 최적화할 수 있습니다.
- 모델은 기본적으로 가장 높은 정밀도와 가장 낮은 성능 설정을 가져야 합니다.
- Diffusers에 이미 있는 모델 아키텍처로 분류할 수 있는 새로운 모델 체크포인트를 통합할 때는 기존 모델 아키텍처를 새로운 체크포인트와 호환되도록 수정해야 합니다. 새로운 파일을 만들어야 하는 경우는 모델 아키텍처가 근본적으로 다른 경우에만 해당합니다.
- 모델은 미래의 변경 사항을 쉽게 확장할 수 있도록 설계되어야 합니다. 이는 공개 함수 인수들과 구성 인수들을 제한하고,미래의 변경 사항을 "예상"하는 것을 통해 달성할 수 있습니다. 예를 들어, 불리언 `is_..._type` 인수보다는 새로운 미래 유형에 쉽게 확장할 수 있는 문자열 "...type" 인수를 추가하는 것이 일반적으로 더 좋습니다. 새로운 모델 체크포인트가 작동하도록 하기 위해 기존 아키텍처에 최소한의 변경만을 가해야 합니다.
- 모델 디자인은 코드의 가독성과 간결성을 유지하는 것과 많은 모델 체크포인트를 지원하는 것 사이의 어려운 균형 조절입니다. 모델링 코드의 대부분은 새로운 모델 체크포인트를 위해 클래스를 수정하는 것이 좋지만, [UNet 블록](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/unet_2d_blocks.py) 및 [Attention 프로세서](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py)와 같이 코드를 장기적으로 간결하고 읽기 쉽게 유지하기 위해 새로운 클래스를 추가하는 예외도 있습니다.

### 스케줄러 [[schedulers]]

스케줄러는 추론을 위한 노이즈 제거 과정을 안내하고 훈련을 위한 노이즈 스케줄을 정의하는 역할을 합니다. 스케줄러는 개별 클래스로 설계되어 있으며, 로드 가능한 구성 파일과 **단일 파일 정책**을 엄격히 따릅니다.

다음과 같은 설계 원칙을 따릅니다:
- 모든 스케줄러는 [`src/diffusers/schedulers`](https://github.com/huggingface/diffusers/tree/main/src/diffusers/schedulers)에서 찾을 수 있습니다.
- 스케줄러는 큰 유틸리티 파일에서 가져오지 **않아야** 하며, 자체 포함성을 유지해야 합니다.
- 하나의 스케줄러 Python 파일은 하나의 스케줄러 알고리즘(논문에서 정의된 것과 같은)에 해당합니다.
- 스케줄러가 유사한 기능을 공유하는 경우, `# Copied from` 메커니즘을 사용할 수 있습니다.
- 모든 스케줄러는 `SchedulerMixin`과 `ConfigMixin`을 상속합니다.
- [`ConfigMixin.from_config`](https://huggingface.co/docs/diffusers/main/en/api/configuration#diffusers.ConfigMixin.from_config) 메소드를 사용하여 스케줄러를 쉽게 교체할 수 있습니다. 자세한 내용은 [여기](../using-diffusers/schedulers.md)에서 설명합니다.
- 모든 스케줄러는 `set_num_inference_steps`와 `step` 함수를 가져야 합니다. `set_num_inference_steps(...)`는 각 노이즈 제거 과정(즉, `step(...)`이 호출되기 전) 이전에 호출되어야 합니다.
- 각 스케줄러는 모델이 호출될 타임스텝의 배열인 `timesteps` 속성을 통해 루프를 돌 수 있는 타임스텝을 노출합니다.
- `step(...)` 함수는 예측된 모델 출력과 "현재" 샘플(x_t)을 입력으로 받고, "이전" 약간 더 노이즈가 제거된 샘플(x_t-1)을 반환합니다.
- 노이즈 제거 스케줄러의 복잡성을 고려하여, `step` 함수는 모든 복잡성을 노출하지 않으며, "블랙 박스"일 수 있습니다.
- 거의 모든 경우에 새로운 스케줄러는 새로운 스케줄링 파일에 구현되어야 합니다.
