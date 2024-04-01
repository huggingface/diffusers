<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# 다양한 Stable Diffusion 포맷 불러오기

Stable Diffusion 모델들은 학습 및 저장된 프레임워크와 다운로드 위치에 따라 다양한 형식으로 제공됩니다. 이러한 형식을 🤗 Diffusers에서 사용할 수 있도록 변환하면 추론을 위한 [다양한 스케줄러 사용](schedulers), 사용자 지정 파이프라인 구축, 추론 속도 최적화를 위한 다양한 기법과 방법 등 라이브러리에서 지원하는 모든 기능을 사용할 수 있습니다.

<Tip>

우리는 `.safetensors` 형식을 추천합니다. 왜냐하면 기존의 pickled 파일은 취약하고 머신에서 코드를 실행할 때 악용될 수 있는 것에 비해 훨씬 더 안전합니다. (safetensors 불러오기 가이드에서 자세히 알아보세요.)

</Tip>

이 가이드에서는 다른 Stable Diffusion 형식을 🤗 Diffusers와 호환되도록 변환하는 방법을 설명합니다.

## PyTorch .ckpt

체크포인트 또는 `.ckpt` 형식은 일반적으로 모델을 저장하는 데 사용됩니다. `.ckpt` 파일은 전체 모델을 포함하며 일반적으로 크기가 몇 GB입니다. `.ckpt` 파일을 [~StableDiffusionPipeline.from_ckpt] 메서드를 사용하여 직접 불러와서 사용할 수도 있지만, 일반적으로 두 가지 형식을 모두 사용할 수 있도록 `.ckpt` 파일을 🤗 Diffusers로 변환하는 것이 더 좋습니다.

`.ckpt` 파일을 변환하는 두 가지 옵션이 있습니다. Space를 사용하여 체크포인트를 변환하거나 스크립트를 사용하여 `.ckpt` 파일을 변환합니다.

### Space로 변환하기

`.ckpt` 파일을 변환하는 가장 쉽고 편리한 방법은 SD에서 Diffusers로 스페이스를 사용하는 것입니다. Space의 지침에 따라 .ckpt 파일을 변환 할 수 있습니다.

이 접근 방식은 기본 모델에서는 잘 작동하지만 더 많은 사용자 정의 모델에서는 어려움을 겪을 수 있습니다. 빈 pull request나 오류를 반환하면 Space가 실패한 것입니다.
이 경우 스크립트를 사용하여 `.ckpt` 파일을 변환해 볼 수 있습니다.

### 스크립트로 변환하기

🤗 Diffusers는 `.ckpt`  파일 변환을 위한 변환 스크립트를 제공합니다. 이 접근 방식은 위의 Space보다 더 안정적입니다.

시작하기 전에 스크립트를 실행할 🤗 Diffusers의 로컬 클론(clone)이 있는지 확인하고 Hugging Face 계정에 로그인하여 pull request를 열고 변환된 모델을 허브에 푸시할 수 있도록 하세요.

```bash
huggingface-cli login
```

스크립트를 사용하려면:

1. 변환하려는 `.ckpt`  파일이 포함된 리포지토리를 Git으로 클론(clone)합니다.

이 예제에서는 TemporalNet .ckpt 파일을 변환해 보겠습니다:

```bash
git lfs install
git clone https://huggingface.co/CiaraRowles/TemporalNet
```

2. 체크포인트를 변환할 리포지토리에서 pull request를 엽니다:

```bash
cd TemporalNet && git fetch origin refs/pr/13:pr/13
git checkout pr/13
```

3. 변환 스크립트에서 구성할 입력 인수는 여러 가지가 있지만 가장 중요한 인수는 다음과 같습니다:

- `checkpoint_path`: 변환할 `.ckpt` 파일의 경로를 입력합니다.
- `original_config_file`: 원래 아키텍처의 구성을 정의하는 YAML 파일입니다. 이 파일을 찾을 수 없는 경우 `.ckpt` 파일을 찾은 GitHub 리포지토리에서 YAML 파일을 검색해 보세요.
- `dump_path`: 변환된 모델의 경로

예를 들어, TemporalNet 모델은 Stable Diffusion v1.5 및 ControlNet 모델이기 때문에 ControlNet 리포지토리에서 cldm_v15.yaml 파일을 가져올 수 있습니다.

4. 이제 스크립트를 실행하여 .ckpt 파일을 변환할 수 있습니다:

```bash
python ../diffusers/scripts/convert_original_stable_diffusion_to_diffusers.py --checkpoint_path temporalnetv3.ckpt --original_config_file cldm_v15.yaml --dump_path ./ --controlnet
```

5. 변환이 완료되면 변환된 모델을 업로드하고 결과물을 pull request [pull request](https://huggingface.co/CiaraRowles/TemporalNet/discussions/13)를 테스트하세요!

```bash
git push origin pr/13:refs/pr/13
```

## **Keras .pb or .h5**

🧪 이 기능은 실험적인 기능입니다. 현재로서는 Stable Diffusion v1 체크포인트만 변환 KerasCV Space에서 지원됩니다.

[KerasCV](https://keras.io/keras_cv/)는 [Stable Diffusion](https://github.com/keras-team/keras-cv/blob/master/keras_cv/models/stable_diffusion)  v1 및 v2에 대한 학습을 지원합니다. 그러나 추론 및 배포를 위한 Stable Diffusion 모델 실험을 제한적으로 지원하는 반면, 🤗 Diffusers는 다양한 [noise schedulers](https://huggingface.co/docs/diffusers/using-diffusers/schedulers), [flash attention](https://huggingface.co/docs/diffusers/optimization/xformers), and [other optimization techniques](https://huggingface.co/docs/diffusers/optimization/fp16) 등 이러한 목적을 위한 보다 완벽한 기능을 갖추고 있습니다.

[Convert KerasCV](https://huggingface.co/spaces/sayakpaul/convert-kerascv-sd-diffusers) Space 변환은 `.pb` 또는 `.h5`을 PyTorch로 변환한 다음, 추론할 수 있도록 [`StableDiffusionPipeline`] 으로 감싸서 준비합니다. 변환된 체크포인트는 Hugging Face Hub의 리포지토리에 저장됩니다.

예제로, textual-inversion으로 학습된 `[sayakpaul/textual-inversion-kerasio](https://huggingface.co/sayakpaul/textual-inversion-kerasio/tree/main)` 체크포인트를 변환해 보겠습니다. 이것은 특수 토큰  `<my-funny-cat>`을 사용하여 고양이로 이미지를 개인화합니다.

KerasCV Space 변환에서는 다음을 입력할 수 있습니다:

- Hugging Face 토큰.
- UNet 과 텍스트 인코더(text encoder) 가중치를 다운로드하는 경로입니다. 모델을 어떻게 학습할지 방식에 따라, UNet과 텍스트 인코더의 경로를 모두 제공할 필요는 없습니다. 예를 들어, textual-inversion에는 텍스트 인코더의 임베딩만 필요하고 텍스트-이미지(text-to-image) 모델 변환에는 UNet 가중치만 필요합니다.
- Placeholder 토큰은 textual-inversion 모델에만 적용됩니다.
- `output_repo_prefix`는 변환된 모델이 저장되는 리포지토리의 이름입니다.

**Submit** (제출) 버튼을 클릭하면 KerasCV 체크포인트가 자동으로 변환됩니다! 체크포인트가 성공적으로 변환되면, 변환된 체크포인트가 포함된 새 리포지토리로 연결되는 링크가 표시됩니다. 새 리포지토리로 연결되는 링크를 따라가면 변환된 모델을 사용해 볼 수 있는 추론 위젯이 포함된 모델 카드가 생성된 KerasCV Space 변환을 확인할 수 있습니다.

코드를 사용하여 추론을 실행하려면 모델 카드의 오른쪽 상단 모서리에 있는 **Use in Diffusers**  버튼을 클릭하여 예시 코드를 복사하여 붙여넣습니다:

```py
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained("sayakpaul/textual-inversion-cat-kerascv_sd_diffusers_pipeline")
```

그러면 다음과 같은 이미지를 생성할 수 있습니다:

```py
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained("sayakpaul/textual-inversion-cat-kerascv_sd_diffusers_pipeline")
pipeline.to("cuda")

placeholder_token = "<my-funny-cat-token>"
prompt = f"two {placeholder_token} getting married, photorealistic, high quality"
image = pipeline(prompt, num_inference_steps=50).images[0]
```

## **A1111 LoRA files**

[Automatic1111](https://github.com/AUTOMATIC1111/stable-diffusion-webui) (A1111)은 Stable Diffusion을 위해 널리 사용되는 웹 UI로, [Civitai](https://civitai.com/) 와 같은 모델 공유 플랫폼을 지원합니다. 특히 LoRA 기법으로 학습된 모델은 학습 속도가 빠르고 완전히 파인튜닝된 모델보다 파일 크기가 훨씬 작기 때문에 인기가 높습니다.

🤗 Diffusers는 [`~loaders.LoraLoaderMixin.load_lora_weights`]:를 사용하여 A1111 LoRA 체크포인트 불러오기를 지원합니다:

```py
from diffusers import DiffusionPipeline, UniPCMultistepScheduler
import torch

pipeline = DiffusionPipeline.from_pretrained(
    "andite/anything-v4.0", torch_dtype=torch.float16, safety_checker=None
).to("cuda")
pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
```

Civitai에서 LoRA 체크포인트를 다운로드하세요; 이 예제에서는  [Howls Moving Castle,Interior/Scenery LoRA (Ghibli Stlye)](https://civitai.com/models/14605?modelVersionId=19998) 체크포인트를 사용했지만, 어떤 LoRA 체크포인트든 자유롭게 사용해 보세요!

```bash
!wget https://civitai.com/api/download/models/19998 -O howls_moving_castle.safetensors
```

메서드를 사용하여 파이프라인에 LoRA 체크포인트를 불러옵니다:

```py
pipeline.load_lora_weights(".", weight_name="howls_moving_castle.safetensors")
```

이제 파이프라인을 사용하여 이미지를 생성할 수 있습니다:

```py
prompt = "masterpiece, illustration, ultra-detailed, cityscape, san francisco, golden gate bridge, california, bay area, in the snow, beautiful detailed starry sky"
negative_prompt = "lowres, cropped, worst quality, low quality, normal quality, artifacts, signature, watermark, username, blurry, more than one bridge, bad architecture"

images = pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=512,
    height=512,
    num_inference_steps=25,
    num_images_per_prompt=4,
    generator=torch.manual_seed(0),
).images
```

마지막으로, 디스플레이에 이미지를 표시하는 헬퍼 함수를 만듭니다:

```py
from PIL import Image


def image_grid(imgs, rows=2, cols=2):
    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


image_grid(images)
```

<div class="flex justify-center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/a1111-lora-sf.png" />
</div>
