<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# 설치

사용하시는 라이브러리에 맞는 🤗 Diffusers를 설치하세요.

🤗 Diffusers는 Python 3.8+, PyTorch 1.7.0+ 및 flax에서 테스트되었습니다. 사용중인 딥러닝 라이브러리에 대한 아래의 설치 안내를 따르세요.

- [PyTorch 설치 안내](https://pytorch.org/get-started/locally/)
- [Flax 설치 안내](https://flax.readthedocs.io/en/latest/)

## pip를 이용한 설치

[가상 환경](https://docs.python.org/3/library/venv.html)에 🤗 Diffusers를 설치해야 합니다.
Python 가상 환경에 익숙하지 않은 경우 [가상환경 pip 설치 가이드](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/)를 살펴보세요.
가상 환경을 사용하면 서로 다른 프로젝트를 더 쉽게 관리하고, 종속성간의 호환성 문제를 피할 수 있습니다.

프로젝트 디렉토리에 가상 환경을 생성하는 것으로 시작하세요:

```bash
python -m venv .env
```

그리고 가상 환경을 활성화합니다:

```bash
source .env/bin/activate
```

이제 다음의 명령어로 🤗 Diffusers를 설치할 준비가 되었습니다:

**PyTorch의 경우**

```bash
pip install diffusers["torch"]
```

**Flax의 경우**

```bash
pip install diffusers["flax"]
```

## 소스로부터 설치

소스에서 `diffusers`를 설치하기 전에, `torch` 및 `accelerate`이 설치되어 있는지 확인하세요.

`torch` 설치에 대해서는 [torch docs](https://pytorch.org/get-started/locally/#start-locally)를 참고하세요.

다음과 같이 `accelerate`을 설치하세요.

```bash
pip install accelerate
```

다음 명령어를 사용하여 소스에서 🤗 Diffusers를 설치하세요:

```bash
pip install git+https://github.com/huggingface/diffusers
```

이 명령어는 최신 `stable` 버전이 아닌 최첨단 `main` 버전을 설치합니다.
`main` 버전은 최신 개발 정보를 최신 상태로 유지하는 데 유용합니다.
예를 들어 마지막 공식 릴리즈 이후 버그가 수정되었지만, 새 릴리즈가 아직 출시되지 않은 경우입니다.
그러나 이는 `main` 버전이 항상 안정적이지 않을 수 있음을 의미합니다.
우리는 `main` 버전이 지속적으로 작동하도록 노력하고 있으며, 대부분의 문제는 보통 몇 시간 또는 하루 안에 해결됩니다.
문제가 발생하면 더 빨리 해결할 수 있도록 [Issue](https://github.com/huggingface/transformers/issues)를 열어주세요!


## 편집가능한 설치

다음을 수행하려면 편집가능한 설치가 필요합니다:

* 소스 코드의 `main` 버전을 사용
* 🤗 Diffusers에 기여 (코드의 변경 사항을 테스트하기 위해 필요)

저장소를 복제하고 다음 명령어를 사용하여 🤗 Diffusers를 설치합니다:

```bash
git clone https://github.com/huggingface/diffusers.git
cd diffusers
```

**PyTorch의 경우**

```
pip install -e ".[torch]"
```

**Flax의 경우**

```
pip install -e ".[flax]"
```

이러한 명령어들은 저장소를 복제한 폴더와 Python 라이브러리 경로를 연결합니다.
Python은 이제 일반 라이브러리 경로에 더하여 복제한 폴더 내부를 살펴봅니다.
예를들어 Python 패키지가 `~/anaconda3/envs/main/lib/python3.8/site-packages/`에 설치되어 있는 경우 Python은 복제한 폴더인 `~/diffusers/`도 검색합니다.

<Tip warning={true}>

라이브러리를 계속 사용하려면 `diffusers` 폴더를 유지해야 합니다.

</Tip>

이제 다음 명령어를 사용하여 최신 버전의 🤗 Diffusers로 쉽게 업데이트할 수 있습니다:

```bash
cd ~/diffusers/
git pull
```

이렇게 하면, 다음에 실행할 때 Python 환경이 🤗 Diffusers의 `main` 버전을 찾게 됩니다.

## 텔레메트리 로깅에 대한 알림

우리 라이브러리는 `from_pretrained()` 요청 중에 텔레메트리 정보를 원격으로 수집합니다.
이 데이터에는 Diffusers 및 PyTorch/Flax의 버전, 요청된 모델 또는 파이프라인 클래스, 그리고 허브에서 호스팅되는 경우 사전학습된 체크포인트에 대한 경로를 포함합니다.
이 사용 데이터는 문제를 디버깅하고 새로운 기능의 우선순위를 지정하는데 도움이 됩니다.
텔레메트리는 HuggingFace 허브에서 모델과 파이프라인을 불러올 때만 전송되며, 로컬 사용 중에는 수집되지 않습니다.

우리는 추가 정보를 공유하지 않기를 원하는 사람이 있다는 것을 이해하고 개인 정보를 존중하므로, 터미널에서 `DISABLE_TELEMETRY` 환경 변수를 설정하여 텔레메트리 수집을 비활성화할 수 있습니다.

Linux/MacOS에서:
```bash
export DISABLE_TELEMETRY=YES
```

Windows에서:
```bash
set DISABLE_TELEMETRY=YES
```