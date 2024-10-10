<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Core ML로 Stable Diffusion을 실행하는 방법

[Core ML](https://developer.apple.com/documentation/coreml)은 Apple 프레임워크에서 지원하는 모델 형식 및 머신 러닝 라이브러리입니다. macOS 또는 iOS/iPadOS 앱 내에서 Stable Diffusion 모델을 실행하는 데 관심이 있는 경우, 이 가이드에서는 기존 PyTorch 체크포인트를 Core ML 형식으로 변환하고 이를 Python 또는 Swift로 추론에 사용하는 방법을 설명합니다.

Core ML 모델은 Apple 기기에서 사용할 수 있는 모든 컴퓨팅 엔진들, 즉 CPU, GPU, Apple Neural Engine(또는 Apple Silicon Mac 및 최신 iPhone/iPad에서 사용할 수 있는 텐서 최적화 가속기인 ANE)을 활용할 수 있습니다. 모델과 실행 중인 기기에 따라 Core ML은 컴퓨팅 엔진도 혼합하여 사용할 수 있으므로, 예를 들어 모델의 일부가 CPU에서 실행되는 반면 다른 부분은 GPU에서 실행될 수 있습니다.

<Tip>

PyTorch에 내장된 `mps` 가속기를 사용하여 Apple Silicon Macs에서 `diffusers` Python 코드베이스를 실행할 수도 있습니다. 이 방법은 [mps 가이드]에 자세히 설명되어 있지만 네이티브 앱과 호환되지 않습니다.

</Tip>

## Stable Diffusion Core ML 체크포인트

Stable Diffusion 가중치(또는 체크포인트)는 PyTorch 형식으로 저장되기 때문에 네이티브 앱에서 사용하기 위해서는 Core ML 형식으로 변환해야 합니다.

다행히도 Apple 엔지니어들이 `diffusers`를 기반으로 한 [변환 툴](https://github.com/apple/ml-stable-diffusion#-converting-models-to-core-ml)을 개발하여 PyTorch 체크포인트를 Core ML로 변환할 수 있습니다.

모델을 변환하기 전에 잠시 시간을 내어 Hugging Face Hub를 살펴보세요. 관심 있는 모델이 이미 Core ML 형식으로 제공되고 있을 가능성이 높습니다:

- [Apple](https://huggingface.co/apple) organization에는 Stable Diffusion 버전 1.4, 1.5, 2.0 base 및 2.1 base가 포함되어 있습니다.
- [coreml](https://huggingface.co/coreml) organization에는 커스텀 DreamBooth가 적용되거나, 파인튜닝된 모델이 포함되어 있습니다.
- 이 [필터](https://huggingface.co/models?pipeline_tag=text-to-image&library=coreml&p=2&sort=likes)를 사용하여 사용 가능한 모든 Core ML 체크포인트들을 반환합니다.

원하는 모델을 찾을 수 없는 경우 Apple의 [모델을 Core ML로 변환하기](https://github.com/apple/ml-stable-diffusion#-converting-models-to-core-ml) 지침을 따르는 것이 좋습니다.

## 사용할 Core ML 변형(Variant) 선택하기

Stable Diffusion 모델은 다양한 목적에 따라 다른 Core ML 변형으로 변환할 수 있습니다:

- 사용되는 어텐션 블록 유형. 어텐션 연산은 이미지 표현의 여러 영역 간의 관계에 '주의를 기울이고' 이미지와 텍스트 표현이 어떻게 연관되어 있는지 이해하는 데 사용됩니다. 어텐션 연산은 컴퓨팅 및 메모리 집약적이므로 다양한 장치의 하드웨어 특성을 고려한 다양한 구현이 존재합니다. Core ML Stable Diffusion 모델의 경우 두 가지 주의 변형이 있습니다:
    * `split_einsum` ([Apple에서 도입](https://machinelearning.apple.com/research/neural-engine-transformers)은 최신 iPhone, iPad 및 M 시리즈 컴퓨터에서 사용할 수 있는 ANE 장치에 최적화되어 있습니다.
    * "원본" 어텐션(`diffusers`에 사용되는 기본 구현)는 CPU/GPU와만 호환되며 ANE와는 호환되지 않습니다. "원본" 어텐션을 사용하여 CPU + GPU에서 모델을 실행하는 것이 ANE보다 *더* 빠를 수 있습니다. 자세한 내용은 [이 성능 벤치마크](https://huggingface.co/blog/fast-mac-diffusers#performance-benchmarks)와 커뮤니티에서 제공하는 일부 [추가 측정](https://github.com/huggingface/swift-coreml-diffusers/issues/31)을 참조하십시오.

- 지원되는 추론 프레임워크
    * `packages`는 Python 추론에 적합합니다. 네이티브 앱에 통합하기 전에 변환된 Core ML 모델을 테스트하거나, Core ML 성능을 알고 싶지만 네이티브 앱을 지원할 필요는 없는 경우에 사용할 수 있습니다. 예를 들어, 웹 UI가 있는 애플리케이션은 Python Core ML 백엔드를 완벽하게 사용할 수 있습니다.
    * Swift 코드에는 `컴파일된` 모델이 필요합니다. Hub의 `컴파일된` 모델은 iOS 및 iPadOS 기기와의 호환성을 위해 큰 UNet 모델 가중치를 여러 파일로 분할합니다. 이는 [`--chunk-unet` 변환 옵션](https://github.com/apple/ml-stable-diffusion#-converting-models-to-core-ml)에 해당합니다. 네이티브 앱을 지원하려면 `컴파일된` 변형을 선택해야 합니다.

공식 Core ML Stable Diffusion [모델](https://huggingface.co/apple/coreml-stable-diffusion-v1-4/tree/main)에는 이러한 변형이 포함되어 있지만 커뮤니티 버전은 다를 수 있습니다:

```
coreml-stable-diffusion-v1-4
├── README.md
├── original
│   ├── compiled
│   └── packages
└── split_einsum
    ├── compiled
    └── packages
```

아래와 같이 필요한 변형을 다운로드하여 사용할 수 있습니다.

## Python에서 Core ML 추론

Python에서 Core ML 추론을 실행하려면 다음 라이브러리를 설치하세요:

```bash
pip install huggingface_hub
pip install git+https://github.com/apple/ml-stable-diffusion
```

### 모델 체크포인트 다운로드하기

`컴파일된` 버전은 Swift와만 호환되므로 Python에서 추론을 실행하려면 `packages` 폴더에 저장된 버전 중 하나를 사용하세요. `원본` 또는 `split_einsum` 어텐션 중 어느 것을 사용할지 선택할 수 있습니다.

다음은 Hub에서 'models'라는 디렉토리로 'original' 어텐션 변형을 다운로드하는 방법입니다:

```Python
from huggingface_hub import snapshot_download
from pathlib import Path

repo_id = "apple/coreml-stable-diffusion-v1-4"
variant = "original/packages"

model_path = Path("./models") / (repo_id.split("/")[-1] + "_" + variant.replace("/", "_"))
snapshot_download(repo_id, allow_patterns=f"{variant}/*", local_dir=model_path, local_dir_use_symlinks=False)
print(f"Model downloaded at {model_path}")
```


### 추론[[python-inference]]

모델의 snapshot을 다운로드한 후에는 Apple의 Python 스크립트를 사용하여 테스트할 수 있습니다.

```shell
python -m python_coreml_stable_diffusion.pipeline --prompt "a photo of an astronaut riding a horse on mars" -i models/coreml-stable-diffusion-v1-4_original_packages -o </path/to/output/image> --compute-unit CPU_AND_GPU --seed 93
```

`<output-mlpackages-directory>`는 위 단계에서 다운로드한 체크포인트를 가리켜야 하며, `--compute-unit`은 추론을 허용할 하드웨어를 나타냅니다. 이는 다음 옵션 중 하나이어야 합니다: `ALL`, `CPU_AND_GPU`, `CPU_ONLY`, `CPU_AND_NE`. 선택적 출력 경로와 재현성을 위한 시드를 제공할 수도 있습니다.

추론 스크립트에서는 Stable Diffusion 모델의 원래 버전인 `CompVis/stable-diffusion-v1-4`를 사용한다고 가정합니다. 다른 모델을 사용하는 경우 추론 명령줄에서 `--model-version` 옵션을 사용하여 해당 허브 ID를 *지정*해야 합니다. 이는 이미 지원되는 모델과 사용자가 직접 학습하거나 파인튜닝한 사용자 지정 모델에 적용됩니다.

예를 들어, [`stable-diffusion-v1-5/stable-diffusion-v1-5`](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)를 사용하려는 경우입니다:

```shell
python -m python_coreml_stable_diffusion.pipeline --prompt "a photo of an astronaut riding a horse on mars" --compute-unit ALL -o output --seed 93 -i models/coreml-stable-diffusion-v1-5_original_packages --model-version stable-diffusion-v1-5/stable-diffusion-v1-5
```


## Swift에서 Core ML 추론하기

Swift에서 추론을 실행하는 것은 모델이 이미 `mlmodelc` 형식으로 컴파일되어 있기 때문에 Python보다 약간 빠릅니다. 이는 앱이 시작될 때 모델이 불러와지는 것이 눈에 띄지만, 이후 여러 번 실행하면 눈에 띄지 않을 것입니다.

### 다운로드

Mac에서 Swift에서 추론을 실행하려면 `컴파일된` 체크포인트 버전 중 하나가 필요합니다. 이전 예제와 유사하지만 `컴파일된` 변형 중 하나를 사용하여 Python 코드를 로컬로 다운로드하는 것이 좋습니다:

```Python
from huggingface_hub import snapshot_download
from pathlib import Path

repo_id = "apple/coreml-stable-diffusion-v1-4"
variant = "original/compiled"

model_path = Path("./models") / (repo_id.split("/")[-1] + "_" + variant.replace("/", "_"))
snapshot_download(repo_id, allow_patterns=f"{variant}/*", local_dir=model_path, local_dir_use_symlinks=False)
print(f"Model downloaded at {model_path}")
```

### 추론[[swift-inference]]

추론을 실행하기 위해서, Apple의 리포지토리를 복제하세요:

```bash
git clone https://github.com/apple/ml-stable-diffusion
cd ml-stable-diffusion
```

그 다음 Apple의 명령어 도구인 [Swift 패키지 관리자](https://www.swift.org/package-manager/#)를 사용합니다:

```bash
swift run StableDiffusionSample --resource-path models/coreml-stable-diffusion-v1-4_original_compiled --compute-units all "a photo of an astronaut riding a horse on mars"
```

`--resource-path`에 이전 단계에서 다운로드한 체크포인트 중 하나를 지정해야 하므로 확장자가 `.mlmodelc`인 컴파일된 Core ML 번들이 포함되어 있는지 확인하시기 바랍니다. `--compute-units`는 다음 값 중 하나이어야 합니다: `all`, `cpuOnly`, `cpuAndGPU`, `cpuAndNeuralEngine`.

자세한 내용은 [Apple의 리포지토리 안의 지침](https://github.com/apple/ml-stable-diffusion)을 참고하시기 바랍니다.


## 지원되는 Diffusers 기능

Core ML 모델과 추론 코드는 🧨 Diffusers의 많은 기능, 옵션 및 유연성을 지원하지 않습니다. 다음은 유의해야 할 몇 가지 제한 사항입니다:

- Core ML 모델은 추론에만 적합합니다. 학습이나 파인튜닝에는 사용할 수 없습니다.
- Swift에 포팅된 스케줄러는 Stable Diffusion에서 사용하는 기본 스케줄러와 `diffusers` 구현에서 Swift로 포팅한 `DPMSolverMultistepScheduler` 두 개뿐입니다. 이들 중 약 절반의 스텝으로 동일한 품질을 생성하는 `DPMSolverMultistepScheduler`를 사용하는 것이 좋습니다.
- 추론 코드에서 네거티브 프롬프트, classifier-free guidance scale 및 image-to-image 작업을 사용할 수 있습니다. depth guidance, ControlNet, latent upscalers와 같은 고급 기능은 아직 사용할 수 없습니다.

Apple의 [변환 및 추론 리포지토리](https://github.com/apple/ml-stable-diffusion)와 자체 [swift-coreml-diffusers](https://github.com/huggingface/swift-coreml-diffusers) 리포지토리는 다른 개발자들이 구축할 수 있는 기술적인 데모입니다.

누락된 기능이 있다고 생각되면 언제든지 기능을 요청하거나, 더 좋은 방법은 기여 PR을 열어주세요. :)


## 네이티브 Diffusers Swift 앱

자체 Apple 하드웨어에서 Stable Diffusion을 실행하는 쉬운 방법 중 하나는 `diffusers`와 Apple의 변환 및 추론 리포지토리를 기반으로 하는 [자체 오픈 소스 Swift 리포지토리](https://github.com/huggingface/swift-coreml-diffusers)를 사용하는 것입니다. 코드를 공부하고 [Xcode](https://developer.apple.com/xcode/)로 컴파일하여 필요에 맞게 조정할 수 있습니다. 편의를 위해 앱스토어에 [독립형 Mac 앱](https://apps.apple.com/app/diffusers/id1666309574)도 있으므로 코드나 IDE를 다루지 않고도 사용할 수 있습니다. 개발자로서 Core ML이 Stable Diffusion 앱을 구축하는 데 가장 적합한 솔루션이라고 판단했다면, 이 가이드의 나머지 부분을 사용하여 프로젝트를 시작할 수 있습니다. 여러분이 무엇을 빌드할지 기대됩니다. :)