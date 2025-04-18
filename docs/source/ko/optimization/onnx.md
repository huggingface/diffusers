<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->


# 추론을 위해 ONNX 런타임을 사용하는 방법

🤗 Diffusers는 ONNX Runtime과 호환되는 Stable Diffusion 파이프라인을 제공합니다. 이를 통해 ONNX(CPU 포함)를 지원하고 PyTorch의 가속 버전을 사용할 수 없는 모든 하드웨어에서 Stable Diffusion을 실행할 수 있습니다.

## 설치

다음 명령어로 ONNX Runtime를 지원하는 🤗 Optimum를 설치합니다:

```sh
pip install optimum["onnxruntime"]
```

## Stable Diffusion 추론

아래 코드는 ONNX 런타임을 사용하는 방법을 보여줍니다. `StableDiffusionPipeline` 대신 `OnnxStableDiffusionPipeline`을 사용해야 합니다.
PyTorch 모델을 불러오고 즉시 ONNX 형식으로 변환하려는 경우 `export=True`로 설정합니다.

```python
from optimum.onnxruntime import ORTStableDiffusionPipeline

model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
pipe = ORTStableDiffusionPipeline.from_pretrained(model_id, export=True)
prompt = "a photo of an astronaut riding a horse on mars"
images = pipe(prompt).images[0]
pipe.save_pretrained("./onnx-stable-diffusion-v1-5")
```

파이프라인을 ONNX 형식으로 오프라인으로 내보내고 나중에 추론에 사용하려는 경우,
[`optimum-cli export`](https://huggingface.co/docs/optimum/main/en/exporters/onnx/usage_guides/export_a_model#exporting-a-model-to-onnx-using-the-cli) 명령어를 사용할 수 있습니다:

```bash
optimum-cli export onnx --model stable-diffusion-v1-5/stable-diffusion-v1-5 sd_v15_onnx/
```

그 다음 추론을 수행합니다:

```python
from optimum.onnxruntime import ORTStableDiffusionPipeline

model_id = "sd_v15_onnx"
pipe = ORTStableDiffusionPipeline.from_pretrained(model_id)
prompt = "a photo of an astronaut riding a horse on mars"
images = pipe(prompt).images[0]
```

Notice that we didn't have to specify `export=True` above.

[Optimum 문서](https://huggingface.co/docs/optimum/)에서 더 많은 예시를 찾을 수 있습니다.

## 알려진 이슈들

- 여러 프롬프트를 배치로 생성하면 너무 많은 메모리가 사용되는 것 같습니다. 이를 조사하는 동안, 배치 대신 반복 방법이 필요할 수도 있습니다.
