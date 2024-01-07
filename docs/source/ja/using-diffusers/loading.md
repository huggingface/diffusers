<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# パイプライン、モデル、スケジューラの読み込み

[[open-in-colab]]

拡散システムを推論に使用する簡単な方法があることは、🧨 Diffusers にとって必要不可欠です。拡散システムは多くの場合、パラメータ化されたモデル、トークナイザ、スケジューラのような複数のコンポーネントから構成され、それらは複雑な方法で相互作用しています。そのため、拡散システム全体の複雑さを使いやすいAPIで包み込むように、私たちは [`DiffusionPipeline`] を設計しました。その一方で、各コンポーネントを個別に読み込んで、独自の拡散システムを構築するなどの他のユースケースにも適応できるような柔軟性も保っています。

推論や学習に必要なものは全て `from_pretrained()` メソッドでアクセスできます。

このガイドでは、以下のコンポーネントの読み込み方を示します:

- Hub やローカルからのパイプライン
- パイプラインへの様々なコンポーネント
- 異なる浮動小数点精度や非指数平均 (non-EMA) 重みのチェックポイントの種類
- モデルやスケジューラ

## 拡散パイプライン

<Tip>

💡 [`DiffusionPipeline`] クラスがどのように動作するのかを詳しく知りたい場合は、[DiffusionPipeline explained](#diffusionpipeline-explained) のセクションまでスキップしてください。

</Tip>

[`DiffusionPipeline`] クラスは、[Hub](https://huggingface.co/models?library=diffusers&sort=trending) から最新のトレンドの拡散モデルを読み込むための最もシンプルで汎用的なクラスです。[`DiffusionPipeline.from_pretrained`] メソッドは、チェックポイントから適切なパイプラインを自動的に検出し、必要な全ての設定ファイルと重みをダウンロードしてキャッシュし、推論のためのインスタンスを返します。

```python
from diffusers import DiffusionPipeline

repo_id = "runwayml/stable-diffusion-v1-5"
pipe = DiffusionPipeline.from_pretrained(repo_id, use_safetensors=True)
```

特定のパイプラインクラスを用いてチェックポイントを読み込むこともできます。上の例では Stable Diffusion モデルを読み込みましたが、同じ結果を得るには [`StableDiffusionPipeline`] クラスを使うこともできます。 

```python
from diffusers import StableDiffusionPipeline

repo_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(repo_id, use_safetensors=True)
```

チェックポイント ([`CompVis/stable-diffusion-v1-4`](https://huggingface.co/CompVis/stable-diffusion-v1-4) や [`runwayml/stable-diffusion-v1-5`](https://huggingface.co/runwayml/stable-diffusion-v1-5)) は、text-to-image や image-to-image のように、複数のタスクに使用することもできます。チェックポイントをどのタスクに使用するかを区別するには、対応するタスク固有のパイプラインクラスを用いてチェックポイントを読み込む必要があります。

```python
from diffusers import StableDiffusionImg2ImgPipeline

repo_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(repo_id)
```

### ローカルのパイプライン

ローカルに拡散パイプラインを読み込むには、 [`git-lfs`](https://git-lfs.github.com/) を用いてチェックポイント (この場合、[`runwayml/stable-diffusion-v1-5`](https://huggingface.co/runwayml/stable-diffusion-v1-5)) をローカルディスクに手動でダウンロードします。これにより、ディスク上に `./stable-diffusion-v1-5` というローカルフォルダが作成されます。 

```bash
git-lfs install
git clone https://huggingface.co/runwayml/stable-diffusion-v1-5
```

そして、ローカルパスを [`~DiffusionPipeline.from_pretrained`] に渡しましょう:

```python
from diffusers import DiffusionPipeline

repo_id = "./stable-diffusion-v1-5"
stable_diffusion = DiffusionPipeline.from_pretrained(repo_id, use_safetensors=True)
```

[`~DiffusionPipeline.from_pretrained`] メソッドは、ローカルパスを検出した際に Hub からファイルをダウンロードすることはありませんが、これはチェックポイントの最新の変更をダウンロードしてキャッシュしないことも意味しています。

### パイプライン中のコンポーネントを入れ替える

パイプラインのデフォルトのコンポーネントは、互換性のある他のコンポーネントでカスタマイズすることができます。カスタマイズが重要なことには、次のような理由があります。

- スケジューラを変更することは、生成速度と品質のトレードオフを探る上で重要。
- モデルの異なるコンポーネントは、通常独立して学習され、より性能の良いコンポーネントと交換可能。
- ファインチューニングでは、通常 UNet やテキストエンコーダなどの一部のコンポーネントだけを学習。

どのスケジューラがカスタマイズに対応しているかを調べるには、`compatibles` メソッドを使います:

```py
from diffusers import DiffusionPipeline

repo_id = "runwayml/stable-diffusion-v1-5"
stable_diffusion = DiffusionPipeline.from_pretrained(repo_id, use_safetensors=True)
stable_diffusion.scheduler.compatibles
```

ここでは、[`SchedulerMixin.from_pretrained`] メソッドを使用して、デフォルトの [`PNDMScheduler`] をよりパフォーマンスの高いスケジューラである [`EulerDiscreteScheduler`] に置き換えてみましょう。`subfolder="scheduler"` 引数は、パイプラインのリポジトリの正しい [subfolder](https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main/scheduler) からスケジューラの設定を読み込むために必要です。

そして、新しい [`EulerDiscreteScheduler`] インスタンスを [`DiffusionPipeline`] の `scheduler` 引数に渡します:

```python
from diffusers import DiffusionPipeline, EulerDiscreteScheduler

repo_id = "runwayml/stable-diffusion-v1-5"
scheduler = EulerDiscreteScheduler.from_pretrained(repo_id, subfolder="scheduler")
stable_diffusion = DiffusionPipeline.from_pretrained(repo_id, scheduler=scheduler, use_safetensors=True)
```

### セーフティチェッカー

Stable Diffusion のような拡散モデルは有害なコンテンツを生成する可能性があります。そのため、🧨 Diffusers には [safety checker](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/safety_checker.py) があり、生成された出力を既知のハードコーディングされた NSFW コンテンツと比較してチェックします。なんらかの理由でセーフティチェッカーを無効化したい場合は、`safety_checker` 引数に `None` を渡してください:

```python
from diffusers import DiffusionPipeline

repo_id = "runwayml/stable-diffusion-v1-5"
stable_diffusion = DiffusionPipeline.from_pretrained(repo_id, safety_checker=None, use_safetensors=True)
"""
あなたは `safety_checker=None` を渡すことで、<class 'diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline'> のセーフティチェッカーを無効化しました。Stable Diffusion のライセンスを遵守し、フィルタリングされていない結果を一般公開されているサービスやアプリケーションで公開しないようにしてください。diffusers チームと Hugging Face は一般公開される全ての状況でセーフティフィルタを有効にしておくことを強く推奨します。詳細については、https://github.com/huggingface/diffusers/pull/254 をご確認ください。
"""
```

### パイプライン間でコンポーネントを再利用する

また、同じコンポーネントを複数のパイプラインで再利用することで、重みを RAM 上で二度読み込むことを避けることができます。コンポーネントを保存するには、[`~DiffusionPipeline.components`] メソッドを使います:

```python
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline

model_id = "runwayml/stable-diffusion-v1-5"
stable_diffusion_txt2img = StableDiffusionPipeline.from_pretrained(model_id, use_safetensors=True)

components = stable_diffusion_txt2img.components
```

そうすることで、重みを RAM に再読み込みすることなく、`components` を別のパイプラインに渡すことができます:

```py
stable_diffusion_img2img = StableDiffusionImg2ImgPipeline(**components)
```

どのコンポーネントを再利用するか、または無効にするかをより柔軟に設定したい場合は、コンポーネントを個別にパイプラインに渡すこともできます。たとえば、text-to-image パイプラインで使用したコンポーネントのうち、セーフティチェッカーと特徴抽出器以外のコンポーネントを image-to-image パイプラインで再利用する場合などです:

```py
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline

model_id = "runwayml/stable-diffusion-v1-5"
stable_diffusion_txt2img = StableDiffusionPipeline.from_pretrained(model_id, use_safetensors=True)
stable_diffusion_img2img = StableDiffusionImg2ImgPipeline(
    vae=stable_diffusion_txt2img.vae,
    text_encoder=stable_diffusion_txt2img.text_encoder,
    tokenizer=stable_diffusion_txt2img.tokenizer,
    unet=stable_diffusion_txt2img.unet,
    scheduler=stable_diffusion_txt2img.scheduler,
    safety_checker=None,
    feature_extractor=None,
    requires_safety_checker=False,
)
```

## チェックポイントの種類

チェックポイントの種類 (checkpoint variants) には、通常以下のようなものがあります:

- [`torch.float16`](https://pytorch.org/docs/stable/tensors.html#data-types) のように、ダウンロードに必要な帯域幅とストレージが半分で済むために、低精度かつ低容量な異なる浮動小数点型で保存されるもの。学習を続けている場合や、CPU を利用している場合は、使用することができません。
- 非指数平均 (non-EMA) 重みのように、推論で使うべきではないが、モデルのファインチューニングを続けるために使用するべきもの。

<Tip>

💡 チェックポイントが同一のモデル構造を持っている一方で、異なるデータセットと設定で学習された場合は別々のリポジトリに保存する必要があります。(例えば、[`stable-diffusion-v1-4`] と [`stable-diffusion-v1-5`])

</Tip>

それ以外では、バリアント (variants) はオリジナルのチェックポイントと **同一** です。これらは全く同じシリアライズフォーマット ([Safetensors](./using_safetensors) など)、同じモデル構造、同じテンソルの形状の重みを持ちます。

| **チェックポイントのタイプ** | **重みの名前**                           | **重みを読み込む際の引数**          |
|------------------|-------------------------------------|--------------------------|
| original         | diffusion_pytorch_model.bin         |                          |
| floating point   | diffusion_pytorch_model.fp16.bin    | `variant`, `torch_dtype` |
| non-EMA          | diffusion_pytorch_model.non_ema.bin | `variant`                |

バリアント (variants) を読み込む際には、2つの重要な引数がある:

- `torch_dtype` は、読み込むチェックポイントの浮動小数点精度を指定します。例えば、`fp16` のバリアントを読み込んで帯域幅を節約したい場合には、`torch_dtype=torch.float16` を指定して `fp16` に**重みを変換**する必要があります。そうしない場合、`fp16` の重みはデフォルトの `fp32` の精度に変換されます。また、`variant` 引数を指定せずにオリジナルのチェックポイントを読み込み、`torch_dtype=torch.float16` で `fp16` に変換することもできます。この場合、デフォルトの `fp32` の重みが最初にダウンロードされ、読み込まれた後に `fp16` に変換されます。

- `variant` は、どのファイルをリポジトリから読み込むかを指定します。例えば、[`diffusers/stable-diffusion-variants`](https://huggingface.co/diffusers/stable-diffusion-variants/tree/main/unet) リポジトリから `non_ema` バリアントを読み込みたい場合、`variant="non_ema"` を指定して `non_ema` ファイルをダウンロードする必要があります。

```python
from diffusers import DiffusionPipeline
import torch

# fp16 バリアントを読み込む
stable_diffusion = DiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", variant="fp16", torch_dtype=torch.float16, use_safetensors=True
)
# non_ema バリアントを読み込む
stable_diffusion = DiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", variant="non_ema", use_safetensors=True
)
```

異なる浮動小数点型や non-EMA バリアントで保存されたチェックポイントを保存するには、[`DiffusionPipeline.save_pretrained`] メソッドを使用し、引数に `variant` を指定します。 元のチェックポイントと同じフォルダバリアントを保存するようにすると、同じフォルダから両方のチェックポイントを読み込むことができます:

```python
from diffusers import DiffusionPipeline

# fp16 バリアントとして保存
stable_diffusion.save_pretrained("runwayml/stable-diffusion-v1-5", variant="fp16")
# non-ema バリアントとして保存
stable_diffusion.save_pretrained("runwayml/stable-diffusion-v1-5", variant="non_ema")
```

バリアントを既存のフォルダに保存しない場合は、`variant` 引数を指定しなければいけません。そうしないと、元のチェックポイントが見つからずに `Exception` が発生します。

```python
# 👎 これはダメ
stable_diffusion = DiffusionPipeline.from_pretrained(
    "./stable-diffusion-v1-5", torch_dtype=torch.float16, use_safetensors=True
)
# 👍 こっちは良い
stable_diffusion = DiffusionPipeline.from_pretrained(
    "./stable-diffusion-v1-5", variant="fp16", torch_dtype=torch.float16, use_safetensors=True
)
```

<!--
TODO(Patrick) - Make sure to uncomment this part as soon as things are deprecated.

#### Using `revision` to load pipeline variants is deprecated

Previously the `revision` argument of [`DiffusionPipeline.from_pretrained`] was heavily used to
load model variants, e.g.:

```python
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", revision="fp16", use_safetensors=True)
```

However, this behavior is now deprecated since the "revision" argument should (just as it's done in GitHub) better be used to load model checkpoints from a specific commit or branch in development.

The above example is therefore deprecated and won't be supported anymore for `diffusers >= 1.0.0`.

<Tip warning={true}>

If you load diffusers pipelines or models with `revision="fp16"` or `revision="non_ema"`,
please make sure to update the code and use `variant="fp16"` or `variation="non_ema"` respectively
instead.

</Tip>
-->

## モデル

モデルは [`ModelMixin.from_pretrained`] メソッドによって読み込まれ、最新バージョンのモデルの重みと設定をダウンロードしてキャッシュします。最新のファイルがローカルキャッシュに存在する場合、[`~ModelMixin.from_pretrained`] は再ダウンロードする代わりにキャッシュ内のファイルを再利用します。

モデルは `subfolder` 引数を用いてサブフォルダから読み込むことができます。たとえば、`runwayml/stable-diffusion-v1-5` のモデルの重みは、[`unet`](https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main/unet) サブフォルダに格納されています:

```python
from diffusers import UNet2DConditionModel

repo_id = "runwayml/stable-diffusion-v1-5"
model = UNet2DConditionModel.from_pretrained(repo_id, subfolder="unet", use_safetensors=True)
```

もしくは、直接リポジトリの [ディレクトリ](https://huggingface.co/google/ddpm-cifar10-32/tree/main) から次のように読み込むことができます:

```python
from diffusers import UNet2DModel

repo_id = "google/ddpm-cifar10-32"
model = UNet2DModel.from_pretrained(repo_id, use_safetensors=True)
```

また、[`ModelMixin.from_pretrained`] と [`ModelMixin.save_pretrained`] において `variant` 引数を指定することで、モデルのバリアントを読み込んだり保存したりすることができます:

```python
from diffusers import UNet2DConditionModel

model = UNet2DConditionModel.from_pretrained(
    "runwayml/stable-diffusion-v1-5", subfolder="unet", variant="non_ema", use_safetensors=True
)
model.save_pretrained("./local-unet", variant="non_ema")
```

## スケジューラ

スケジューラは [`SchedulerMixin.from_pretrained`] メソッドによって読み込まれます。スケジューラはモデルとは異なり、**パラメータ化**されていたり、**学習** されていません。

スケジューラを読み込んでも、メモリを大量に消費することはなく、同じ設定ファイルを様々なスケジューラに使用することができます。
たとえば、以下のスケジューラは [`StableDiffusionPipeline`] と互換性があり、これらのクラスは同じスケジューラの設定ファイルで読み込むことができます。

```python
from diffusers import StableDiffusionPipeline
from diffusers import (
    DDPMScheduler,
    DDIMScheduler,
    PNDMScheduler,
    LMSDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    DPMSolverMultistepScheduler,
)

repo_id = "runwayml/stable-diffusion-v1-5"

ddpm = DDPMScheduler.from_pretrained(repo_id, subfolder="scheduler")
ddim = DDIMScheduler.from_pretrained(repo_id, subfolder="scheduler")
pndm = PNDMScheduler.from_pretrained(repo_id, subfolder="scheduler")
lms = LMSDiscreteScheduler.from_pretrained(repo_id, subfolder="scheduler")
euler_anc = EulerAncestralDiscreteScheduler.from_pretrained(repo_id, subfolder="scheduler")
euler = EulerDiscreteScheduler.from_pretrained(repo_id, subfolder="scheduler")
dpm = DPMSolverMultistepScheduler.from_pretrained(repo_id, subfolder="scheduler")

# `dpm` を `ddpm`、`ddim`、`pndm`、`lms`、`euler_anc`、`euler` のいずれかに置き換える。
pipeline = StableDiffusionPipeline.from_pretrained(repo_id, scheduler=dpm, use_safetensors=True)
```

## DiffusionPipeline の説明

[`DiffusionPipeline.from_pretrained`] はクラスメソッドとして、次の2つの処理を行います:

- 推論に必要なフォルダ構造の最新版をダウンロードしてキャッシュします。最新のフォルダ構造が既にローカルキャッシュに存在する場合は、[`DiffusionPipeline.from_pretrained`] はキャッシュを再利用し、ファイルを再ダウンロードすることはありません。
- キャッシュされた重みを `model_index.json` から得られた適切なパイプライン[クラス](../api/pipelines/overview#diffusers-summary) に読み込み、そのインスタンスを返します。

パイプラインの基本的なフォルダ構造は、そのクラスのインスタンスに直接対応しています。たとえば、[`StableDiffusionPipeline`] は [`runwayml/stable-diffusion-v1-5`] のフォルダ構造に対応しています。

```python
from diffusers import DiffusionPipeline

repo_id = "runwayml/stable-diffusion-v1-5"
pipeline = DiffusionPipeline.from_pretrained(repo_id, use_safetensors=True)
print(pipeline)
```

ここでは、パイプラインが [`StableDiffusionPipeline`] のインスタンスであり、7つのコンポーネントから構成されていることがわかります:

- `"feature_extractor"`: 🤗 Transformers の [`~transformers.CLIPImageProcessor`]
- `"safety_checker"`: 有害なコンテンツをスクリーニングするための [コンポーネント](https://github.com/huggingface/diffusers/blob/e55687e1e15407f60f32242027b7bb8170e58266/src/diffusers/pipelines/stable_diffusion/safety_checker.py#L32)
- `"scheduler"`: [`PNDMScheduler`] のインスタンス
- `"text_encoder"`: 🤗 Transformers の [`~transformers.CLIPTextModel`]
- `"tokenizer"`: 🤗 Transformers の [`~transformers.CLIPTokenizer`]
- `"unet"`: [`UNet2DConditionModel`] のインスタンス
- `"vae"`: [`AutoencoderKL`] のインスタンス

```json
StableDiffusionPipeline {
  "feature_extractor": [
    "transformers",
    "CLIPImageProcessor"
  ],
  "safety_checker": [
    "stable_diffusion",
    "StableDiffusionSafetyChecker"
  ],
  "scheduler": [
    "diffusers",
    "PNDMScheduler"
  ],
  "text_encoder": [
    "transformers",
    "CLIPTextModel"
  ],
  "tokenizer": [
    "transformers",
    "CLIPTokenizer"
  ],
  "unet": [
    "diffusers",
    "UNet2DConditionModel"
  ],
  "vae": [
    "diffusers",
    "AutoencoderKL"
  ]
}
```

パイプラインのインスタンスに含まれるコンポーネントを、[`runwayml/stable-diffusion-v1-5`](https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main) のフォルダ構造と比較してみましょう。リポジトリ内のコンポーネントごとに別々のフォルダがあることがわかります:

```
.
├── feature_extractor
│   └── preprocessor_config.json
├── model_index.json
├── safety_checker
│   ├── config.json
|   ├── model.fp16.safetensors
│   ├── model.safetensors
│   ├── pytorch_model.bin
|   └── pytorch_model.fp16.bin
├── scheduler
│   └── scheduler_config.json
├── text_encoder
│   ├── config.json
|   ├── model.fp16.safetensors
│   ├── model.safetensors
│   |── pytorch_model.bin
|   └── pytorch_model.fp16.bin
├── tokenizer
│   ├── merges.txt
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   └── vocab.json
├── unet
│   ├── config.json
│   ├── diffusion_pytorch_model.bin
|   |── diffusion_pytorch_model.fp16.bin
│   |── diffusion_pytorch_model.f16.safetensors
│   |── diffusion_pytorch_model.non_ema.bin
│   |── diffusion_pytorch_model.non_ema.safetensors
│   └── diffusion_pytorch_model.safetensors
|── vae
.   ├── config.json
.   ├── diffusion_pytorch_model.bin
    ├── diffusion_pytorch_model.fp16.bin
    ├── diffusion_pytorch_model.fp16.safetensors
    └── diffusion_pytorch_model.safetensors
```

パイプラインの各コンポーネントに属性としてアクセスし、その設定を見ることができます:

```py
pipeline.tokenizer
CLIPTokenizer(
    name_or_path="/root/.cache/huggingface/hub/models--runwayml--stable-diffusion-v1-5/snapshots/39593d5650112b4cc580433f6b0435385882d819/tokenizer",
    vocab_size=49408,
    model_max_length=77,
    is_fast=False,
    padding_side="right",
    truncation_side="right",
    special_tokens={
        "bos_token": AddedToken("<|startoftext|>", rstrip=False, lstrip=False, single_word=False, normalized=True),
        "eos_token": AddedToken("<|endoftext|>", rstrip=False, lstrip=False, single_word=False, normalized=True),
        "unk_token": AddedToken("<|endoftext|>", rstrip=False, lstrip=False, single_word=False, normalized=True),
        "pad_token": "<|endoftext|>",
    },
    clean_up_tokenization_spaces=True
)
```

すべてのパイプラインは、[`DiffusionPipeline`] に次のような情報を伝えるための [`model_index.json`](https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/model_index.json) ファイルが必要です:

- `_class_name` より、どのパイプラインクラスをロードするか
- `_diffusers_version` より、どのバージョンの 🧨 Diffusers を使ってモデルが作成されたか
- どのライブラリのどのコンポーネントがサブフォルダに格納されているか (ここで、`name` はコンポーネント名とサブフォルダ名、`library` はクラスを読み込むライブラリ名、`class` はクラス名に対応します)

```json
{
  "_class_name": "StableDiffusionPipeline",
  "_diffusers_version": "0.6.0",
  "feature_extractor": [
    "transformers",
    "CLIPImageProcessor"
  ],
  "safety_checker": [
    "stable_diffusion",
    "StableDiffusionSafetyChecker"
  ],
  "scheduler": [
    "diffusers",
    "PNDMScheduler"
  ],
  "text_encoder": [
    "transformers",
    "CLIPTextModel"
  ],
  "tokenizer": [
    "transformers",
    "CLIPTokenizer"
  ],
  "unet": [
    "diffusers",
    "UNet2DConditionModel"
  ],
  "vae": [
    "diffusers",
    "AutoencoderKL"
  ]
}
```
