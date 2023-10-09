# 학습을 위한 데이터셋 만들기

[Hub](https://huggingface.co/datasets?task_categories=task_categories:text-to-image&sort=downloads) 에는 모델 교육을 위한 많은 데이터셋이 있지만,
관심이 있거나 사용하고 싶은 데이터셋을 찾을 수 없는 경우 🤗 [Datasets](hf.co/docs/datasets) 라이브러리를 사용하여 데이터셋을 만들 수 있습니다.
데이터셋 구조는 모델을 학습하려는 작업에 따라 달라집니다.
가장 기본적인 데이터셋 구조는 unconditional 이미지 생성과 같은 작업을 위한 이미지 디렉토리입니다.
또 다른 데이터셋 구조는 이미지 디렉토리와 text-to-image 생성과 같은 작업에 해당하는 텍스트 캡션이 포함된 텍스트 파일일 수 있습니다.

이 가이드에는 파인 튜닝할 데이터셋을 만드는 두 가지 방법을 소개합니다:

- 이미지 폴더를 `--train_data_dir` 인수에 제공합니다.
- 데이터셋을 Hub에 업로드하고 데이터셋 리포지토리 id를 `--dataset_name` 인수에 전달합니다.

<Tip>

💡 학습에 사용할 이미지 데이터셋을 만드는 방법에 대한 자세한 내용은 [이미지 데이터셋 만들기](https://huggingface.co/docs/datasets/image_dataset) 가이드를 참고하세요.

</Tip>

## 폴더 형태로 데이터셋 구축하기

Unconditional 생성을 위해 이미지 폴더로 자신의 데이터셋을 구축할 수 있습니다.
학습 스크립트는 🤗 Datasets의 [ImageFolder](https://huggingface.co/docs/datasets/en/image_dataset#imagefolder) 빌더를 사용하여
자동으로 폴더에서 데이터셋을 구축합니다. 디렉토리 구조는 다음과 같아야 합니다 :

```bash
data_dir/xxx.png
data_dir/xxy.png
data_dir/[...]/xxz.png
```

데이터셋 디렉터리의 경로를 `--train_data_dir` 인수로 전달한 다음 학습을 시작할 수 있습니다:

```bash
accelerate launch train_unconditional.py \
    # argument로 폴더 지정하기 \
    --train_data_dir <path-to-train-directory> \
    <other-arguments>
```

## Hub에 데이터 올리기

<Tip>

💡 데이터셋을 만들고 Hub에 업로드하는 것에 대한 자세한 내용은 [🤗 Datasets을 사용한 이미지 검색](https://huggingface.co/blog/image-search-datasets) 게시물을 참고하세요.

</Tip>

PIL 인코딩된 이미지가 포함된 `이미지` 열을 생성하는 [이미지 폴더](https://huggingface.co/docs/datasets/image_load#imagefolder) 기능을 사용하여 데이터셋 생성을 시작합니다.

`data_dir` 또는 `data_files` 매개 변수를 사용하여 데이터셋의 위치를 지정할 수 있습니다.
`data_files` 매개변수는 특정 파일을 `train` 이나 `test` 로 분리한 데이터셋에 매핑하는 것을 지원합니다:

```python
from datasets import load_dataset

# 예시 1: 로컬 폴더
dataset = load_dataset("imagefolder", data_dir="path_to_your_folder")

# 예시 2: 로컬 파일 (지원 포맷 : tar, gzip, zip, xz, rar, zstd)
dataset = load_dataset("imagefolder", data_files="path_to_zip_file")

# 예시 3: 원격 파일 (지원 포맷 : tar, gzip, zip, xz, rar, zstd)
dataset = load_dataset(
    "imagefolder",
    data_files="https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip",
)

# 예시 4: 여러개로 분할
dataset = load_dataset(
    "imagefolder", data_files={"train": ["path/to/file1", "path/to/file2"], "test": ["path/to/file3", "path/to/file4"]}
)
```

[push_to_hub(https://huggingface.co/docs/datasets/v2.13.1/en/package_reference/main_classes#datasets.Dataset.push_to_hub) 을 사용해서 Hub에 데이터셋을 업로드 합니다:

```python
# 터미널에서 huggingface-cli login 커맨드를 이미 실행했다고 가정합니다
dataset.push_to_hub("name_of_your_dataset")

# 개인 repo로 push 하고 싶다면, `private=True` 을 추가하세요:
dataset.push_to_hub("name_of_your_dataset", private=True)
```

이제 데이터셋 이름을 `--dataset_name` 인수에 전달하여 데이터셋을 학습에 사용할 수 있습니다:

```bash
accelerate launch --mixed_precision="fp16"  train_text_to_image.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --dataset_name="name_of_your_dataset" \
  <other-arguments>
```

## 다음 단계

데이터셋을 생성했으니 이제 학습 스크립트의 `train_data_dir` (데이터셋이 로컬이면) 혹은 `dataset_name` (Hub에 데이터셋을 올렸으면) 인수에 연결할 수 있습니다.

다음 단계에서는 데이터셋을 사용하여 [unconditional 생성](https://huggingface.co/docs/diffusers/v0.18.2/en/training/unconditional_training) 또는 [텍스트-이미지 생성](https://huggingface.co/docs/diffusers/training/text2image)을 위한 모델을 학습시켜보세요!
