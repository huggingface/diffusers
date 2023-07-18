<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# 🧨 Diffusers 학습 예시

이번 챕터에서는 다양한 유즈케이스들에 대한 예제 코드들을 통해 어떻게하면 효과적으로 `diffusers` 라이브러리를 사용할 수 있을까에 대해 알아보도록 하겠습니다. 

**Note**: 혹시 오피셜한 예시코드를 찾고 있다면, [여기](https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines)를 참고해보세요!

여기서 다룰 예시들은 다음을 지향합니다.

- **손쉬운 디펜던시 설치** (Self-contained) : 여기서 사용될 예시 코드들의 디펜던시 패키지들은 전부 `pip install` 명령어를 통해 설치 가능한 패키지들입니다. 또한 친절하게 `requirements.txt` 파일에 해당 패키지들이 명시되어 있어, `pip install -r requirements.txt`로 간편하게 해당 디펜던시들을 설치할 수 있습니다. 예시: [train_unconditional.py](https://github.com/huggingface/diffusers/blob/main/examples/unconditional_image_generation/train_unconditional.py), [requirements.txt](https://github.com/huggingface/diffusers/blob/main/examples/unconditional_image_generation/requirements.txt) 
- **손쉬운 수정** (Easy-to-tweak) : 저희는 가능하면 많은 유즈 케이스들을 제공하고자 합니다. 하지만 예시는 결국 그저 예시라는 점들 기억해주세요. 여기서 제공되는 예시코드들을 그저 단순히 복사-붙혀넣기하는 식으로는 여러분이 마주한 문제들을 손쉽게 해결할 순 없을 것입니다. 다시 말해 어느 정도는 여러분의 상황과 니즈에 맞춰 코드를 일정 부분 고쳐나가야 할 것입니다. 따라서 대부분의 학습 예시들은 데이터의 전처리 과정과 학습 과정에 대한 코드들을 함께 제공함으로써, 사용자가 니즈에 맞게 손쉬운 수정할 수 있도록 돕고 있습니다.
- **입문자 친화적인** (Beginner-friendly) : 이번 챕터는 diffusion 모델과 `diffusers` 라이브러리에 대한 전반적인 이해를 돕기 위해 작성되었습니다. 따라서 diffusion 모델에 대한 최신 SOTA (state-of-the-art) 방법론들 가운데서도, 입문자에게는 많이 어려울 수 있다고 판단되면, 해당 방법론들은 여기서 다루지 않으려고 합니다.
- **하나의 태스크만 포함할 것**(One-purpose-only): 여기서 다룰 예시들은 하나의 태스크만 포함하고 있어야 합니다. 물론 이미지 초해상화(super-resolution)와 이미지 보정(modification)과 같은 유사한 모델링 프로세스를 갖는 태스크들이 존재하겠지만, 하나의 예제에 하나의 태스크만을 담는 것이 더 이해하기 용이하다고 판단했기 때문입니다.



저희는 diffusion 모델의 대표적인 태스크들을 다루는 공식 예제를 제공하고 있습니다. *공식* 예제는 현재 진행형으로 `diffusers` 관리자들(maintainers)에 의해 관리되고 있습니다. 또한 저희는 앞서 정의한 저희의 철학을 엄격하게 따르고자 노력하고 있습니다. 혹시 여러분께서 이러한 예시가 반드시 필요하다고 생각되신다면, 언제든지 [Feature Request](https://github.com/huggingface/diffusers/issues/new?assignees=&labels=&template=feature_request.md&title=) 혹은 직접 [Pull Request](https://github.com/huggingface/diffusers/compare)를 주시기 바랍니다. 저희는 언제나 환영입니다!

학습 예시들은 다양한 태스크들에 대해 diffusion 모델을 사전학습(pretrain)하거나 파인튜닝(fine-tuning)하는 법을 보여줍니다. 현재 다음과 같은 예제들을 지원하고 있습니다.

- [Unconditional Training](./unconditional_training)
- [Text-to-Image Training](./text2image)
- [Text Inversion](./text_inversion)
- [Dreambooth](./dreambooth)

memory-efficient attention 연산을 수행하기 위해, 가능하면 [xFormers](../optimization/xformers)를 설치해주시기 바랍니다. 이를 통해 학습 속도를 늘리고 메모리에 대한 부담을 줄일 수 있습니다.

| Task | 🤗 Accelerate | 🤗 Datasets | Colab
|---|---|:---:|:---:|
| [**Unconditional Image Generation**](./unconditional_training) | ✅ | ✅ | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/training_example.ipynb)
| [**Text-to-Image fine-tuning**](./text2image) | ✅ | ✅ | 
| [**Textual Inversion**](./text_inversion) | ✅ | - | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/sd_textual_inversion_training.ipynb)
| [**Dreambooth**](./dreambooth) | ✅ | - | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/sd_dreambooth_training.ipynb)
| [**Training with LoRA**](./lora) | ✅ | - | - |
| [**ControlNet**](./controlnet) | ✅ | ✅ | - |
| [**InstructPix2Pix**](./instructpix2pix) | ✅ | ✅ | - |
| [**Custom Diffusion**](./custom_diffusion) | ✅ | ✅ | - |


## 커뮤니티

공식 예제 외에도 **커뮤니티 예제** 역시 제공하고 있습니다. 해당 예제들은 우리의 커뮤니티에 의해 관리됩니다. 커뮤니티 예쩨는 학습 예시나 추론 파이프라인으로 구성될 수 있습니다. 이러한 커뮤니티 예시들의 경우,  앞서 정의했던 철학들을 좀 더 관대하게 적용하고 있습니다. 또한 이러한 커뮤니티 예시들의 경우, 모든 이슈들에 대한 유지보수를 보장할 수는 없습니다.

유용하긴 하지만, 아직은 대중적이지 못하거나 저희의 철학에 부합하지 않는 예제들은 [community examples](https://github.com/huggingface/diffusers/tree/main/examples/community) 폴더에 담기게 됩니다.

**Note**: 커뮤니티 예제는 `diffusers`에 기여(contribution)를 희망하는 분들에게 [아주 좋은 기여 수단](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22)이 될 수 있습니다.

## 주목할 사항들

최신 버전의 예시 코드들의 성공적인 구동을 보장하기 위해서는, 반드시 **소스코드를 통해 `diffusers`를 설치해야 하며,** 해당 예시 코드들이 요구하는 디펜던시들 역시 설치해야 합니다. 이를 위해 새로운 가상 환경을 구축하고 다음의 명령어를 실행해야 합니다.

```bash
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install .
```

그 다음 `cd` 명령어를 통해 해당 예제 디렉토리에 접근해서 다음 명령어를 실행하면 됩니다.

```bash
pip install -r requirements.txt
```