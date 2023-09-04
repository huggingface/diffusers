# 세이프텐서 로드

[safetensors](https://github.com/huggingface/safetensors)는 텐서를 저장하고 로드하기 위한 안전하고 빠른 파일 형식입니다. 일반적으로 PyTorch 모델 가중치는 Python의 [`pickle`](https://docs.python.org/3/library/pickle.html) 유틸리티를 사용하여 `.bin` 파일에 저장되거나 `피클`됩니다. 그러나 `피클`은 안전하지 않으며 피클된 파일에는 실행될 수 있는 악성 코드가 포함될 수 있습니다. 세이프텐서는 `피클`의 안전한 대안으로 모델 가중치를 공유하는 데 이상적입니다.

이 가이드에서는 `.safetensor` 파일을 로드하는 방법과 다른 형식으로 저장된 안정적 확산 모델 가중치를 `.safetensor`로 변환하는 방법을 보여드리겠습니다. 시작하기 전에 세이프텐서가 설치되어 있는지 확인하세요:

```bash
!pip install safetensors
```

['runwayml/stable-diffusion-v1-5`] (https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main) 리포지토리를 보면 `text_encoder`, `unet` 및 `vae` 하위 폴더에 가중치가 `.safetensors` 형식으로 저장되어 있는 것을 볼 수 있습니다. 기본적으로 🤗 디퓨저는 모델 저장소에서 사용할 수 있는 경우 해당 하위 폴더에서 이러한 '.safetensors` 파일을 자동으로 로드합니다.

보다 명시적인 제어를 위해 선택적으로 `사용_세이프텐서=True`를 설정할 수 있습니다(`세이프텐서`가 설치되지 않은 경우 설치하라는 오류 메시지가 표시됨):

```py
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", use_safetensors=True)
```

그러나 모델 가중치가 위의 예시처럼 반드시 별도의 하위 폴더에 저장되는 것은 아닙니다. 모든 가중치가 하나의 '.safetensors` 파일에 저장되는 경우도 있습니다. 이 경우 가중치가 Stable Diffusion 가중치인 경우 [`~diffusers.loaders.FromCkptMixin.from_ckpt`] 메서드를 사용하여 파일을 직접 로드할 수 있습니다:

```py
from diffusers import StableDiffusionPipeline

pipeline = StableDiffusionPipeline.from_ckpt(
    "https://huggingface.co/WarriorMama777/OrangeMixs/blob/main/Models/AbyssOrangeMix/AbyssOrangeMix.safetensors"
)
```

## 세이프텐서로 변환

허브의 모든 가중치를 '.safetensors` 형식으로 사용할 수 있는 것은 아니며, '.bin`으로 저장된 가중치가 있을 수 있습니다. 이 경우 [Convert Space](https://huggingface.co/spaces/diffusers/convert)을 사용하여 가중치를 '.safetensors'로 변환하세요. Convert Space는 피클된 가중치를 다운로드하여 변환한 후 풀 리퀘스트를 열어 허브에 새로 변환된 `.safetensors` 파일을 업로드합니다. 이렇게 하면 피클된 파일에 악성 코드가 포함되어 있는 경우, 안전하지 않은 파일과 의심스러운 피클 가져오기를 탐지하는 [보안 스캐너](https://huggingface.co/docs/hub/security-pickle#hubs-security-scanner)가 있는 허브로 업로드됩니다. - 개별 컴퓨터가 아닌.

개정` 매개변수에 풀 리퀘스트에 대한 참조를 지정하여 새로운 '.safetensors` 가중치가 적용된 모델을 사용할 수 있습니다(허브의 [Check PR](https://huggingface.co/spaces/diffusers/check_pr) 공간에서 테스트할 수도 있음)(예: `refs/pr/22`):

```py
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", revision="refs/pr/22")
```

## 세이프센서를 사용하는 이유는 무엇인가요?

세이프티 센서를 사용하는 데에는 여러 가지 이유가 있습니다:

- 세이프텐서를 사용하는 가장 큰 이유는 안전입니다.오픈 소스 및 모델 배포가 증가함에 따라 다운로드한 모델 가중치에 악성 코드가 포함되어 있지 않다는 것을 신뢰할 수 있는 것이 중요해졌습니다.세이프센서의 현재 헤더 크기는 매우 큰 JSON 파일을 구문 분석하지 못하게 합니다.
- 모델 전환 간의 로딩 속도는 텐서의 제로 카피를 수행하는 세이프텐서를 사용해야 하는 또 다른 이유입니다. 가중치를 CPU(기본값)로 로드하는 경우 '피클'에 비해 특히 빠르며, 가중치를 GPU로 직접 로드하는 경우에도 빠르지는 않더라도 비슷하게 빠릅니다. 모델이 이미 로드된 경우에만 성능 차이를 느낄 수 있으며, 가중치를 다운로드하거나 모델을 처음 로드하는 경우에는 성능 차이를 느끼지 못할 것입니다.

	전체 파이프라인을 로드하는 데 걸리는 시간입니다:

	```py
 from diffusers import StableDiffusionPipeline

 pipeline = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1")
 "Loaded in safetensors 0:00:02.033658"
 "Loaded in PyTorch 0:00:02.663379"
	```

	하지만 실제로 500MB의 모델 가중치를 로드하는 데 걸리는 시간은 얼마 되지 않습니다:

	```bash
	safetensors: 3.4873ms
	PyTorch: 172.7537ms
	```

지연 로딩은 세이프텐서에서도 지원되며, 이는 분산 설정에서 일부 텐서만 로드하는 데 유용합니다. 이 형식을 사용하면 [BLOOM](https://huggingface.co/bigscience/bloom) 모델을 일반 PyTorch 가중치를 사용하여 10분이 걸리던 것을 8개의 GPU에서 45초 만에 로드할 수 있습니다.
