<!--Copyright 2023 Custom Diffusion authors The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# 커스텀 Diffusion 학습 예제 

[커스텀 Diffusion](https://arxiv.org/abs/2212.04488)은 피사체의 이미지 몇 장(4~5장)만 주어지면 Stable Diffusion처럼 text-to-image 모델을 커스터마이징하는 방법입니다.
'train_custom_diffusion.py' 스크립트는 학습 과정을 구현하고 이를 Stable Diffusion에 맞게 조정하는 방법을 보여줍니다.

이 교육 사례는 [Nupur Kumari](https://nupurkmr9.github.io/)가 제공하였습니다. (Custom Diffusion의 저자 중 한명). 

## 로컬에서 PyTorch로 실행하기

### Dependencies 설치하기

스크립트를 실행하기 전에 라이브러리의 학습 dependencies를 설치해야 합니다:

**중요**

예제 스크립트의 최신 버전을 성공적으로 실행하려면 **소스로부터 설치**하는 것을 매우 권장하며, 예제 스크립트를 자주 업데이트하는 만큼 일부 예제별 요구 사항을 설치하고 설치를 최신 상태로 유지하는 것이 좋습니다. 이를 위해 새 가상 환경에서 다음 단계를 실행하세요:


```bash
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install -e .
```

[example folder](https://github.com/huggingface/diffusers/tree/main/examples/custom_diffusion)로 cd하여 이동하세요.

```
cd examples/custom_diffusion
```

이제 실행

```bash
pip install -r requirements.txt
pip install clip-retrieval 
```

그리고 [🤗Accelerate](https://github.com/huggingface/accelerate/) 환경을 초기화:

```bash
accelerate config
```

또는 사용자 환경에 대한 질문에 답하지 않고 기본 가속 구성을 사용하려면 다음과 같이 하세요.

```bash
accelerate config default
```

또는 사용 중인 환경이 대화형 셸을 지원하지 않는 경우(예: jupyter notebook)

```python
from accelerate.utils import write_basic_config

write_basic_config()
```
### 고양이 예제 😺

이제 데이터셋을 가져옵니다. [여기](https://www.cs.cmu.edu/~custom-diffusion/assets/data.zip)에서 데이터셋을 다운로드하고 압축을 풉니다. 직접 데이터셋을 사용하려면 [학습용 데이터셋 생성하기](create_dataset) 가이드를 참고하세요.

또한 'clip-retrieval'을 사용하여 200개의 실제 이미지를 수집하고, regularization으로서 이를 학습 데이터셋의 타겟 이미지와 결합합니다. 이렇게 하면 주어진 타겟 이미지에 대한 과적합을 방지할 수 있습니다. 다음 플래그를 사용하면 `prior_loss_weight=1.`로 `prior_preservation`, `real_prior` regularization을 활성화할 수 있습니다. 
클래스_프롬프트`는 대상 이미지와 동일한 카테고리 이름이어야 합니다. 수집된 실제 이미지에는 `class_prompt`와 유사한 텍스트 캡션이 있습니다. 검색된 이미지는 `class_data_dir`에 저장됩니다. 생성된 이미지를 regularization으로 사용하기 위해 `real_prior`를 비활성화할 수 있습니다. 실제 이미지를 수집하려면 훈련 전에 이 명령을 먼저 사용하십시오. 

```bash
pip install clip-retrieval
python retrieve.py --class_prompt cat --class_data_dir real_reg/samples_cat --num_class_images 200
```

**___참고: [stable-diffusion-2](https://huggingface.co/stabilityai/stable-diffusion-2) 768x768 모델을 사용하는 경우 '해상도'를 768로 변경하세요.___**

스크립트는 모델 체크포인트와 `pytorch_custom_diffusion_weights.bin` 파일을 생성하여 저장소에 저장합니다.

```bash
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export OUTPUT_DIR="path-to-save-model"
export INSTANCE_DIR="./data/cat"

accelerate launch train_custom_diffusion.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --class_data_dir=./real_reg/samples_cat/ \
  --with_prior_preservation --real_prior --prior_loss_weight=1.0 \
  --class_prompt="cat" --num_class_images=200 \
  --instance_prompt="photo of a <new1> cat"  \
  --resolution=512  \
  --train_batch_size=2  \
  --learning_rate=1e-5  \
  --lr_warmup_steps=0 \
  --max_train_steps=250 \
  --scale_lr --hflip  \
  --modifier_token "<new1>" \
  --push_to_hub
```

**더 낮은 VRAM 요구 사항(GPU당 16GB)으로 더 빠르게 훈련하려면 `--enable_xformers_memory_efficient_attention`을 사용하세요. 설치 방법은 [가이드](https://github.com/facebookresearch/xformers)를 따르세요.**

가중치 및 편향(`wandb`)을 사용하여 실험을 추적하고 중간 결과를 저장하려면(강력히 권장합니다) 다음 단계를 따르세요:

* `wandb` 설치: `pip install wandb`.
* 로그인 : `wandb login`. 
* 그런 다음 트레이닝을 시작하는 동안 `validation_prompt`를 지정하고 `report_to`를 `wandb`로 설정합니다. 다음과 같은 관련 인수를 구성할 수도 있습니다:
    * `num_validation_images`
    * `validation_steps`

```bash
accelerate launch train_custom_diffusion.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --class_data_dir=./real_reg/samples_cat/ \
  --with_prior_preservation --real_prior --prior_loss_weight=1.0 \
  --class_prompt="cat" --num_class_images=200 \
  --instance_prompt="photo of a <new1> cat"  \
  --resolution=512  \
  --train_batch_size=2  \
  --learning_rate=1e-5  \
  --lr_warmup_steps=0 \
  --max_train_steps=250 \
  --scale_lr --hflip  \
  --modifier_token "<new1>" \
  --validation_prompt="<new1> cat sitting in a bucket" \
  --report_to="wandb" \
  --push_to_hub
```

다음은 [Weights and Biases page](https://wandb.ai/sayakpaul/custom-diffusion/runs/26ghrcau)의 예시이며, 여러 학습 세부 정보와 함께 중간 결과들을 확인할 수 있습니다.  

`--push_to_hub`를 지정하면 학습된 파라미터가 허깅 페이스 허브의 리포지토리에 푸시됩니다. 다음은 [예제 리포지토리](https://huggingface.co/sayakpaul/custom-diffusion-cat)입니다.

### 멀티 컨셉에 대한 학습 🐱🪵

[this](https://github.com/ShivamShrirao/diffusers/blob/main/examples/dreambooth/train_dreambooth.py)와 유사하게 각 컨셉에 대한 정보가 포함된 [json](https://github.com/adobe-research/custom-diffusion/blob/main/assets/concept_list.json) 파일을 제공합니다.

실제 이미지를 수집하려면 json 파일의 각 컨셉에 대해 이 명령을 실행합니다. 

```bash
pip install clip-retrieval
python retrieve.py --class_prompt {} --class_data_dir {} --num_class_images 200
```

그럼 우리는 학습시킬 준비가 되었습니다!

```bash
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export OUTPUT_DIR="path-to-save-model"

accelerate launch train_custom_diffusion.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --output_dir=$OUTPUT_DIR \
  --concepts_list=./concept_list.json \
  --with_prior_preservation --real_prior --prior_loss_weight=1.0 \
  --resolution=512  \
  --train_batch_size=2  \
  --learning_rate=1e-5  \
  --lr_warmup_steps=0 \
  --max_train_steps=500 \
  --num_class_images=200 \
  --scale_lr --hflip  \
  --modifier_token "<new1>+<new2>" \
  --push_to_hub
```

다음은 [Weights and Biases page](https://wandb.ai/sayakpaul/custom-diffusion/runs/3990tzkg)의 예시이며, 다른 학습 세부 정보와 함께 중간 결과들을 확인할 수 있습니다.

### 사람 얼굴에 대한 학습

사람 얼굴에 대한 파인튜닝을 위해 다음과 같은 설정이 더 효과적이라는 것을 확인했습니다: `learning_rate=5e-6`, `max_train_steps=1000 to 2000`, `freeze_model=crossattn`을 최소 15~20개의 이미지로 설정합니다.

실제 이미지를 수집하려면 훈련 전에 이 명령을 먼저 사용하십시오.

```bash
pip install clip-retrieval
python retrieve.py --class_prompt person --class_data_dir real_reg/samples_person --num_class_images 200
```

이제 학습을 시작하세요!

```bash
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export OUTPUT_DIR="path-to-save-model"
export INSTANCE_DIR="path-to-images"

accelerate launch train_custom_diffusion.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --class_data_dir=./real_reg/samples_person/ \
  --with_prior_preservation --real_prior --prior_loss_weight=1.0 \
  --class_prompt="person" --num_class_images=200 \
  --instance_prompt="photo of a <new1> person"  \
  --resolution=512  \
  --train_batch_size=2  \
  --learning_rate=5e-6  \
  --lr_warmup_steps=0 \
  --max_train_steps=1000 \
  --scale_lr --hflip --noaug \
  --freeze_model crossattn \
  --modifier_token "<new1>" \
  --enable_xformers_memory_efficient_attention \
  --push_to_hub
```

## 추론

위 프롬프트를 사용하여 모델을 학습시킨 후에는 아래 프롬프트를 사용하여 추론을 실행할 수 있습니다. 프롬프트에 'modifier token'(예: 위 예제에서는 \<new1\>)을 반드시 포함해야 합니다.

```python
import torch
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16).to("cuda")
pipe.unet.load_attn_procs("path-to-save-model", weight_name="pytorch_custom_diffusion_weights.bin")
pipe.load_textual_inversion("path-to-save-model", weight_name="<new1>.bin")

image = pipe(
    "<new1> cat sitting in a bucket",
    num_inference_steps=100,
    guidance_scale=6.0,
    eta=1.0,
).images[0]
image.save("cat.png")
```

허브 리포지토리에서 이러한 매개변수를 직접 로드할 수 있습니다:

```python
import torch
from huggingface_hub.repocard import RepoCard
from diffusers import DiffusionPipeline

model_id = "sayakpaul/custom-diffusion-cat"
card = RepoCard.load(model_id)
base_model_id = card.data.to_dict()["base_model"]

pipe = DiffusionPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16).to("cuda")
pipe.unet.load_attn_procs(model_id, weight_name="pytorch_custom_diffusion_weights.bin")
pipe.load_textual_inversion(model_id, weight_name="<new1>.bin")

image = pipe(
    "<new1> cat sitting in a bucket",
    num_inference_steps=100,
    guidance_scale=6.0,
    eta=1.0,
).images[0]
image.save("cat.png")
```

다음은 여러 컨셉으로 추론을 수행하는 예제입니다:

```python
import torch
from huggingface_hub.repocard import RepoCard
from diffusers import DiffusionPipeline

model_id = "sayakpaul/custom-diffusion-cat-wooden-pot"
card = RepoCard.load(model_id)
base_model_id = card.data.to_dict()["base_model"]

pipe = DiffusionPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16).to("cuda")
pipe.unet.load_attn_procs(model_id, weight_name="pytorch_custom_diffusion_weights.bin")
pipe.load_textual_inversion(model_id, weight_name="<new1>.bin")
pipe.load_textual_inversion(model_id, weight_name="<new2>.bin")

image = pipe(
    "the <new1> cat sculpture in the style of a <new2> wooden pot",
    num_inference_steps=100,
    guidance_scale=6.0,
    eta=1.0,
).images[0]
image.save("multi-subject.png")
```

여기서 '고양이'와 '나무 냄비'는 여러 컨셉을 말합니다.

### 학습된 체크포인트에서 추론하기

`--checkpointing_steps`  인수를 사용한 경우 학습 과정에서 저장된 전체 체크포인트 중 하나에서 추론을 수행할 수도 있습니다. 

## Grads를 None으로 설정

더 많은 메모리를 절약하려면 스크립트에 `--set_grads_to_none` 인수를 전달하세요. 이렇게 하면 성적이 0이 아닌 없음으로 설정됩니다. 그러나 특정 동작이 변경되므로 문제가 발생하면 이 인수를 제거하세요.

자세한 정보: https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html

## 실험 결과

실험에 대한 자세한 내용은 [당사 웹페이지](https://www.cs.cmu.edu/~custom-diffusion/)를 참조하세요. 