<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# InstructPix2Pix

[InstructPix2Pix](https://arxiv.org/abs/2211.09800)는 text-conditioned diffusion 모델이 한 이미지에 편집을 따를 수 있도록 파인튜닝하는 방법입니다. 이 방법을 사용하여 파인튜닝된 모델은 다음을 입력으로 사용합니다:

<p align="center">
    <img src="https://huggingface.co/datasets/diffusers/docs-images/resolve/main/evaluation_diffusion_models/edit-instruction.png" alt="instructpix2pix-inputs" width=600/>
</p>

출력은 입력 이미지에 편집 지시가 반영된 "수정된" 이미지입니다:

<p align="center">
    <img src="https://huggingface.co/datasets/diffusers/docs-images/resolve/main/output-gs%407-igs%401-steps%4050.png" alt="instructpix2pix-output" width=600/>
</p>

`train_instruct_pix2pix.py` 스크립트([여기](https://github.com/huggingface/diffusers/blob/main/examples/instruct_pix2pix/train_instruct_pix2pix.py)에서 찾을 수 있습니다.)는 학습 절차를 설명하고 Stable Diffusion에 적용할 수 있는 방법을 보여줍니다.


*** `train_instruct_pix2pix.py`는 [원래 구현](https://github.com/timothybrooks/instruct-pix2pix)에 충실하면서 InstructPix2Pix 학습 절차를 구현하고 있지만, [소규모 데이터셋](https://huggingface.co/datasets/fusing/instructpix2pix-1000-samples)에서만 테스트를 했습니다. 이는 최종 결과에 영향을 끼칠 수 있습니다. 더 나은 결과를 위해, 더 큰 데이터셋에서 더 길게 학습하는 것을 권장합니다. [여기](https://huggingface.co/datasets/timbrooks/instructpix2pix-clip-filtered)에서 InstructPix2Pix 학습을 위해 큰 데이터셋을 찾을 수 있습니다.
***

## PyTorch로 로컬에서 실행하기

### 종속성(dependencies) 설치하기

이 스크립트를 실행하기 전에, 라이브러리의 학습 종속성을 설치하세요:

**중요**

최신 버전의 예제 스크립트를 성공적으로 실행하기 위해, **원본으로부터 설치**하는 것과 예제 스크립트를 자주 업데이트하고 예제별 요구사항을 설치하기 때문에 최신 상태로 유지하는 것을 권장합니다. 이를 위해, 새로운 가상 환경에서 다음 스텝을 실행하세요:

```bash
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install -e .
```

cd 명령어로 예제 폴더로 이동하세요.
```bash
cd examples/instruct_pix2pix
```

이제 실행하세요.
```bash
pip install -r requirements.txt
```

그리고 [🤗Accelerate](https://github.com/huggingface/accelerate/) 환경에서 초기화하세요:

```bash
accelerate config
```

혹은 환경에 대한 질문 없이 기본적인 accelerate 구성을 사용하려면 다음을 실행하세요.

```bash
accelerate config default
```

혹은 사용 중인 환경이 notebook과 같은 대화형 쉘은 지원하지 않는 경우는 다음 절차를 따라주세요.

```python
from accelerate.utils import write_basic_config

write_basic_config()
```

### 예시

이전에 언급했듯이, 학습을 위해 [작은 데이터셋](https://huggingface.co/datasets/fusing/instructpix2pix-1000-samples)을 사용할 것입니다. 그 데이터셋은 InstructPix2Pix 논문에서 사용된 [원래의 데이터셋](https://huggingface.co/datasets/timbrooks/instructpix2pix-clip-filtered)보다 작은 버전입니다. 자신의 데이터셋을 사용하기 위해, [학습을 위한 데이터셋 만들기](create_dataset) 가이드를 참고하세요.

`MODEL_NAME` 환경 변수(허브 모델 레포지토리 또는 모델 가중치가 포함된 폴더 경로)를 지정하고 [`pretrained_model_name_or_path`](https://huggingface.co/docs/diffusers/en/api/diffusion_pipeline#diffusers.DiffusionPipeline.from_pretrained.pretrained_model_name_or_path) 인수에 전달합니다. `DATASET_ID`에 데이터셋 이름을 지정해야 합니다:


```bash
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export DATASET_ID="fusing/instructpix2pix-1000-samples"
```

지금, 학습을 실행할 수 있습니다. 스크립트는 레포지토리의 하위 폴더의 모든 구성요소(`feature_extractor`, `scheduler`, `text_encoder`, `unet` 등)를 저장합니다.

```bash
accelerate launch --mixed_precision="fp16" train_instruct_pix2pix.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --dataset_name=$DATASET_ID \
    --enable_xformers_memory_efficient_attention \
    --resolution=256 --random_flip \
    --train_batch_size=4 --gradient_accumulation_steps=4 --gradient_checkpointing \
    --max_train_steps=15000 \
    --checkpointing_steps=5000 --checkpoints_total_limit=1 \
    --learning_rate=5e-05 --max_grad_norm=1 --lr_warmup_steps=0 \
    --conditioning_dropout_prob=0.05 \
    --mixed_precision=fp16 \
    --seed=42 \
    --push_to_hub
```


추가적으로, 가중치와 바이어스를 학습 과정에 모니터링하여 검증 추론을 수행하는 것을 지원합니다. `report_to="wandb"`와 이 기능을 사용할 수 있습니다:

```bash
accelerate launch --mixed_precision="fp16" train_instruct_pix2pix.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --dataset_name=$DATASET_ID \
    --enable_xformers_memory_efficient_attention \
    --resolution=256 --random_flip \
    --train_batch_size=4 --gradient_accumulation_steps=4 --gradient_checkpointing \
    --max_train_steps=15000 \
    --checkpointing_steps=5000 --checkpoints_total_limit=1 \
    --learning_rate=5e-05 --max_grad_norm=1 --lr_warmup_steps=0 \
    --conditioning_dropout_prob=0.05 \
    --mixed_precision=fp16 \
    --val_image_url="https://hf.co/datasets/diffusers/diffusers-images-docs/resolve/main/mountain.png" \
    --validation_prompt="make the mountains snowy" \
    --seed=42 \
    --report_to=wandb \
    --push_to_hub
 ```

모델 디버깅에 유용한 이 평가 방법 권장합니다. 이를 사용하기 위해 `wandb`를 설치하는 것을 주목해주세요. `pip install wandb`로 실행해 `wandb`를 설치할 수 있습니다.

[여기](https://wandb.ai/sayakpaul/instruct-pix2pix/runs/ctr3kovq), 몇 가지 평가 방법과 학습 파라미터를 포함하는 예시를 볼 수 있습니다.

 ***참고: 원본 논문에서, 저자들은 256x256 이미지 해상도로 학습한 모델로 512x512와 같은 더 큰 해상도로 잘 일반화되는 것을 볼 수 있었습니다. 이는 학습에 사용한 큰 데이터셋을 사용했기 때문입니다.***

 ## 다수의 GPU로 학습하기

`accelerate`는 원활한 다수의 GPU로 학습을 가능하게 합니다. `accelerate`로 분산 학습을 실행하는 [여기](https://huggingface.co/docs/accelerate/basic_tutorials/launch) 설명을 따라 해 주시기 바랍니다. 예시의 명령어 입니다:


```bash
accelerate launch --mixed_precision="fp16" --multi_gpu train_instruct_pix2pix.py \
 --pretrained_model_name_or_path=runwayml/stable-diffusion-v1-5 \
 --dataset_name=sayakpaul/instructpix2pix-1000-samples \
 --use_ema \
 --enable_xformers_memory_efficient_attention \
 --resolution=512 --random_flip \
 --train_batch_size=4 --gradient_accumulation_steps=4 --gradient_checkpointing \
 --max_train_steps=15000 \
 --checkpointing_steps=5000 --checkpoints_total_limit=1 \
 --learning_rate=5e-05 --lr_warmup_steps=0 \
 --conditioning_dropout_prob=0.05 \
 --mixed_precision=fp16 \
 --seed=42 \
 --push_to_hub
```

 ## 추론하기

일단 학습이 완료되면, 추론 할 수 있습니다:

 ```python
import PIL
import requests
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline

model_id = "your_model_id"  # <- 이를 수정하세요.
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
generator = torch.Generator("cuda").manual_seed(0)

url = "https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/test_pix2pix_4.png"


def download_image(url):
    image = PIL.Image.open(requests.get(url, stream=True).raw)
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image


image = download_image(url)
prompt = "wipe out the lake"
num_inference_steps = 20
image_guidance_scale = 1.5
guidance_scale = 10

edited_image = pipe(
    prompt,
    image=image,
    num_inference_steps=num_inference_steps,
    image_guidance_scale=image_guidance_scale,
    guidance_scale=guidance_scale,
    generator=generator,
).images[0]
edited_image.save("edited_image.png")
```

학습 스크립트를 사용해 얻은 예시의 모델 레포지토리는 여기 [sayakpaul/instruct-pix2pix](https://huggingface.co/sayakpaul/instruct-pix2pix)에서 확인할 수 있습니다.

성능을 위한 속도와 품질을 제어하기 위해 세 가지 파라미터를 사용하는 것이 좋습니다:

* `num_inference_steps`
* `image_guidance_scale`
* `guidance_scale`

특히, `image_guidance_scale`와 `guidance_scale`는 생성된("수정된") 이미지에서 큰 영향을 미칠 수 있습니다.([여기](https://twitter.com/RisingSayak/status/1628392199196151808?s=20)예시를 참고해주세요.)


만약 InstructPix2Pix 학습 방법을 사용해 몇 가지 흥미로운 방법을 찾고 있다면, 이 블로그 게시물[Instruction-tuning Stable Diffusion with InstructPix2Pix](https://huggingface.co/blog/instruction-tuning-sd)을 확인해주세요.