# Textual inversion

[[open-in-colab]]

[`StableDiffusionPipeline`]은  textual-inversion을 지원하는데, 이는 몇 개의 샘플 이미지만으로 stable diffusion과 같은 모델이 새로운 컨셉을 학습할 수 있도록 하는 기법입니다. 이를 통해 생성된 이미지를 더 잘 제어하고 특정 컨셉에 맞게 모델을 조정할 수 있습니다. 커뮤니티에서 만들어진 컨셉들의 컬렉션은 [Stable Diffusion Conceptualizer](https://huggingface.co/spaces/sd-concepts-library/stable-diffusion-conceptualizer)를 통해 빠르게 사용해볼 수 있습니다.

이 가이드에서는 Stable Diffusion Conceptualizer에서 사전학습한 컨셉을 사용하여 textual-inversion으로 추론을 실행하는 방법을 보여드립니다. textual-inversion으로 모델에 새로운 컨셉을 학습시키는 데 관심이 있으시다면,  [Textual Inversion](./training/text_inversion)  훈련 가이드를 참조하세요.

Hugging Face 계정으로 로그인하세요:

```py
from huggingface_hub import notebook_login

notebook_login()
```

필요한 라이브러리를 불러오고 생성된 이미지를 시각화하기 위한 도우미 함수 `image_grid`를 만듭니다:

```py
import os
import torch

import PIL
from PIL import Image

from diffusers import StableDiffusionPipeline
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid
```

Stable Diffusion과 [Stable Diffusion Conceptualizer](https://huggingface.co/spaces/sd-concepts-library/stable-diffusion-conceptualizer)에서 사전학습된 컨셉을 선택합니다:

```py
pretrained_model_name_or_path = "stable-diffusion-v1-5/stable-diffusion-v1-5"
repo_id_embeds = "sd-concepts-library/cat-toy"
```

이제 파이프라인을 로드하고 사전학습된 컨셉을 파이프라인에 전달할 수 있습니다:

```py
pipeline = StableDiffusionPipeline.from_pretrained(pretrained_model_name_or_path, torch_dtype=torch.float16).to("cuda")

pipeline.load_textual_inversion(repo_id_embeds)
```

특별한 placeholder token '`<cat-toy>`'를 사용하여 사전학습된 컨셉으로 프롬프트를 만들고, 생성할 샘플의 수와 이미지 행의 수를 선택합니다:

```py
prompt = "a grafitti in a favela wall with a <cat-toy> on it"

num_samples = 2
num_rows = 2
```

그런 다음 파이프라인을 실행하고, 생성된 이미지들을 저장합니다. 그리고 처음에 만들었던 도우미 함수 `image_grid`를 사용하여 생성 결과들을 시각화합니다. 이 때 `num_inference_steps`와 `guidance_scale`과 같은 매개 변수들을 조정하여, 이것들이 이미지 품질에 어떠한 영향을 미치는지를 자유롭게 확인해보시기 바랍니다.

```py
all_images = []
for _ in range(num_rows):
    images = pipe(prompt, num_images_per_prompt=num_samples, num_inference_steps=50, guidance_scale=7.5).images
    all_images.extend(images)

grid = image_grid(all_images, num_samples, num_rows)
grid
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/textual_inversion_inference.png">
</div>
