<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

[[open-in-colab]]

# 簡単な案内

拡散モデル(Diffusion Model)は、ランダムな正規分布から段階的にノイズ除去するように学習され、画像や音声などの目的のものを生成できます。これは生成AIに多大な関心を呼び起こしました。インターネット上で拡散によって生成された画像の例を見たことがあるでしょう。🧨 Diffusersは、誰もが拡散モデルに広くアクセスできるようにすることを目的としたライブラリです。

この案内では、開発者または日常的なユーザーに関わらず、🧨 Diffusers を紹介し、素早く目的のものを生成できるようにします！このライブラリには3つの主要コンポーネントがあります:

* [`DiffusionPipeline`]は事前に学習された拡散モデルからサンプルを迅速に生成するために設計された高レベルのエンドツーエンドクラス。
*  拡散システムを作成するためのビルディングブロックとして使用できる、人気のある事前学習された[モデル](./api/models)アーキテクチャとモジュール。
*  多くの異なる[スケジューラ](./api/schedulers/overview) - ノイズがどのようにトレーニングのために加えられるか、そして生成中にどのようにノイズ除去された画像を生成するかを制御するアルゴリズム。

この案内では、[`DiffusionPipeline`]を生成に使用する方法を紹介し、モデルとスケジューラを組み合わせて[`DiffusionPipeline`]の内部で起こっていることを再現する方法を説明します。

<Tip>

この案内は🧨 Diffusers [ノートブック](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/diffusers_intro.ipynb)を簡略化したもので、すぐに使い始めることができます。Diffusers 🧨のゴール、設計哲学、コアAPIの詳細についてもっと知りたい方は、ノートブックをご覧ください！

</Tip>

始める前に必要なライブラリーがすべてインストールされていることを確認してください：

```py
# uncomment to install the necessary libraries in Colab
#!pip install --upgrade diffusers accelerate transformers
```

- [🤗 Accelerate](https://huggingface.co/docs/accelerate/index)生成とトレーニングのためのモデルのロードを高速化します
- [Stable Diffusion](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/overview)ような最も一般的な拡散モデルを実行するには、[🤗 Transformers](https://huggingface.co/docs/transformers/index)が必要です。

## 拡散パイプライン

[`DiffusionPipeline`]は事前学習された拡散システムを生成に使用する最も簡単な方法です。これはモデルとスケジューラを含むエンドツーエンドのシステムです。[`DiffusionPipeline`]は多くの作業／タスクにすぐに使用することができます。また、サポートされているタスクの完全なリストについては[🧨Diffusersの概要](./api/pipelines/overview#diffusers-summary)の表を参照してください。

| **タスク**                     | **説明**                                                                                              | **パイプライン**
|------------------------------|--------------------------------------------------------------------------------------------------------------|-----------------|
| Unconditional Image Generation          | 正規分布から画像生成 | [unconditional_image_generation](./using-diffusers/unconditional_image_generation) |
| Text-Guided Image Generation | 文章から画像生成 | [conditional_image_generation](./using-diffusers/conditional_image_generation) |
| Text-Guided Image-to-Image Translation     | 画像と文章から新たな画像生成 | [img2img](./using-diffusers/img2img) |
| Text-Guided Image-Inpainting          | 画像、マスク、および文章が指定された場合に、画像のマスクされた部分を文章をもとに修復 | [inpaint](./using-diffusers/inpaint) |
| Text-Guided Depth-to-Image Translation | 文章と深度推定によって構造を保持しながら画像生成 | [depth2img](./using-diffusers/depth2img) |

まず、[`DiffusionPipeline`]のインスタンスを作成し、ダウンロードしたいパイプラインのチェックポイントを指定します。
この[`DiffusionPipeline`]はHugging Face Hubに保存されている任意の[チェックポイント](https://huggingface.co/models?library=diffusers&sort=downloads)を使用することができます。
この案内では、[`stable-diffusion-v1-5`](https://huggingface.co/runwayml/stable-diffusion-v1-5)チェックポイントでテキストから画像へ生成します。

<Tip warning={true}>

[Stable Diffusion]モデルについては、モデルを実行する前にまず[ライセンス](https://huggingface.co/spaces/CompVis/stable-diffusion-license)を注意深くお読みください。🧨  Diffusers は、攻撃的または有害なコンテンツを防ぐために [`safety_checker`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/safety_checker.py) を実装していますが、モデルの改良された画像生成機能により、潜在的に有害なコンテンツが生成される可能性があります。

</Tip>

モデルを[`~DiffusionPipeline.from_pretrained`]メソッドでロードします：

```python
>>> from diffusers import DiffusionPipeline

>>> pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", use_safetensors=True)
```
[`DiffusionPipeline`]は全てのモデリング、トークン化、スケジューリングコンポーネントをダウンロードしてキャッシュします。Stable Diffusionパイプラインは[`UNet2DConditionModel`]と[`PNDMScheduler`]などで構成されています：

```py
>>> pipeline
StableDiffusionPipeline {
  "_class_name": "StableDiffusionPipeline",
  "_diffusers_version": "0.13.1",
  ...,
  "scheduler": [
    "diffusers",
    "PNDMScheduler"
  ],
  ...,
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

このモデルはおよそ14億個のパラメータで構成されているため、GPU上でパイプラインを実行することを強く推奨します。
PyTorchと同じように、ジェネレータオブジェクトをGPUに移すことができます：

```python
>>> pipeline.to("cuda")
```

これで、文章を `pipeline` に渡して画像を生成し、ノイズ除去された画像にアクセスできるようになりました。デフォルトでは、画像出力は[`PIL.Image`](https://pillow.readthedocs.io/en/stable/reference/Image.html?highlight=image#the-image-class)オブジェクトでラップされます。

```python
>>> image = pipeline("An image of a squirrel in Picasso style").images[0]
>>> image
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/image_of_squirrel_painting.png"/>
</div>

`save`関数で画像を保存できます:

```python
>>> image.save("image_of_squirrel_painting.png")
```

### ローカルパイプライン

ローカルでパイプラインを使用することもできます。唯一の違いは、最初にウェイトをダウンロードする必要があることです：

```bash
!git lfs install
!git clone https://huggingface.co/runwayml/stable-diffusion-v1-5
```

保存したウェイトをパイプラインにロードします：

```python
>>> pipeline = DiffusionPipeline.from_pretrained("./stable-diffusion-v1-5", use_safetensors=True)
```

これで、上のセクションと同じようにパイプラインを動かすことができます。

### スケジューラの交換

スケジューラーによって、ノイズ除去のスピードや品質のトレードオフが異なります。どれが自分に最適かを知る最善の方法は、実際に試してみることです！Diffusers 🧨の主な機能の1つは、スケジューラを簡単に切り替えることができることです。例えば、デフォルトの[`PNDMScheduler`]を[`EulerDiscreteScheduler`]に置き換えるには、[`~diffusers.ConfigMixin.from_config`]メソッドでロードできます：

```py
>>> from diffusers import EulerDiscreteScheduler

>>> pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", use_safetensors=True)
>>> pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config)
```

新しいスケジューラを使って画像を生成し、その違いに気づくかどうか試してみてください！

次のセクションでは、[`DiffusionPipeline`]を構成するコンポーネント（モデルとスケジューラ）を詳しく見て、これらのコンポーネントを使って猫の画像を生成する方法を学びます。

## モデル

ほとんどのモデルはノイズの多いサンプルを取り、各タイムステップで*残りのノイズ*を予測します（他のモデルは前のサンプルを直接予測するか、速度または[`v-prediction`](https://github.com/huggingface/diffusers/blob/5e5ce13e2f89ac45a0066cb3f369462a3cf1d9ef/src/diffusers/schedulers/scheduling_ddim.py#L110)を予測するように学習します）。モデルを混ぜて他の拡散システムを作ることもできます。

モデルは[`~ModelMixin.from_pretrained`]メソッドで開始されます。このメソッドはモデルをローカルにキャッシュするので、次にモデルをロードするときに高速になります。この案内では、[`UNet2DModel`]をロードします。これは基本的な画像生成モデルであり、猫画像で学習されたチェックポイントを使います：

```py
>>> from diffusers import UNet2DModel

>>> repo_id = "google/ddpm-cat-256"
>>> model = UNet2DModel.from_pretrained(repo_id, use_safetensors=True)
```

モデルのパラメータにアクセスするには、`model.config` を呼び出せます：

```py
>>> model.config
```

モデル構成は🧊凍結🧊されたディクショナリであり、モデル作成後にこれらのパラメー タを変更することはできません。これは意図的なもので、最初にモデル・アーキテクチャを定義するために使用されるパラメータが同じままであることを保証します。他のパラメータは生成中に調整することができます。

最も重要なパラメータは以下の通りです：

* sample_size`: 入力サンプルの高さと幅。
* `in_channels`: 入力サンプルの入力チャンネル数。
* down_block_types` と `up_block_types`: UNet アーキテクチャを作成するために使用されるダウンサンプリングブロックとアップサンプリングブロックのタイプ。
* block_out_channels`: ダウンサンプリングブロックの出力チャンネル数。逆順でアップサンプリングブロックの入力チャンネル数にも使用されます。
* layer_per_block`: 各 UNet ブロックに含まれる ResNet ブロックの数。

このモデルを生成に使用するには、ランダムな画像の形の正規分布を作成します。このモデルは複数のランダムな正規分布を受け取ることができるため`batch`軸を入れます。入力チャンネル数に対応する`channel`軸も必要です。画像の高さと幅に対応する`sample_size`軸を持つ必要があります：

```py
>>> import torch

>>> torch.manual_seed(0)

>>> noisy_sample = torch.randn(1, model.config.in_channels, model.config.sample_size, model.config.sample_size)
>>> noisy_sample.shape
torch.Size([1, 3, 256, 256])
```

画像生成には、ノイズの多い画像と `timestep` をモデルに渡します。`timestep`は入力画像がどの程度ノイズが多いかを示します。これは、モデルが拡散プロセスにおける自分の位置を決定するのに役立ちます。モデルの出力を得るには `sample` メソッドを使用します：

```py
>>> with torch.no_grad():
...     noisy_residual = model(sample=noisy_sample, timestep=2).sample
```

しかし、実際の例を生成するには、ノイズ除去プロセスをガイドするスケジューラが必要です。次のセクションでは、モデルをスケジューラと組み合わせる方法を学びます。

## スケジューラ

スケジューラは、モデルの出力（この場合は `noisy_residual` ）が与えられたときに、ノイズの多いサンプルからノイズの少ないサンプルへの移行を管理します。


<Tip>

🧨 Diffusersは拡散システムを構築するためのツールボックスです。[`DiffusionPipeline`]は事前に構築された拡散システムを使い始めるのに便利な方法ですが、独自のモデルとスケジューラコンポーネントを個別に選択してカスタム拡散システムを構築することもできます。

</Tip>

この案内では、[`DDPMScheduler`]を[`~diffusers.ConfigMixin.from_config`]メソッドでインスタンス化します：

```py
>>> from diffusers import DDPMScheduler

>>> scheduler = DDPMScheduler.from_config(repo_id)
>>> scheduler
DDPMScheduler {
  "_class_name": "DDPMScheduler",
  "_diffusers_version": "0.13.1",
  "beta_end": 0.02,
  "beta_schedule": "linear",
  "beta_start": 0.0001,
  "clip_sample": true,
  "clip_sample_range": 1.0,
  "num_train_timesteps": 1000,
  "prediction_type": "epsilon",
  "trained_betas": null,
  "variance_type": "fixed_small"
}
```

<Tip>

💡 スケジューラがどのようにコンフィギュレーションからインスタンス化されるかに注目してください。モデルとは異なり、スケジューラは学習可能な重みを持たず、パラメーターを持ちません！

</Tip>

最も重要なパラメータは以下の通りです：

* num_train_timesteps`: ノイズ除去処理の長さ、言い換えれば、ランダムな正規分布をデータサンプルに処理するのに必要なタイムステップ数です。
* `beta_schedule`: 生成とトレーニングに使用するノイズスケジュールのタイプ。
* `beta_start` と `beta_end`: ノイズスケジュールの開始値と終了値。

少しノイズの少ない画像を予測するには、スケジューラの [`~diffusers.DDPMScheduler.step`] メソッドに以下を渡します: モデルの出力、`timestep`、現在の `sample`。

```py
>>> less_noisy_sample = scheduler.step(model_output=noisy_residual, timestep=2, sample=noisy_sample).prev_sample
>>> less_noisy_sample.shape
```

`less_noisy_sample`は次の`timestep`に渡すことができ、そこでさらにノイズが少なくなります！

では、すべてをまとめて、ノイズ除去プロセス全体を視覚化してみましょう。

まず、ノイズ除去された画像を後処理して `PIL.Image` として表示する関数を作成します：

```py
>>> import PIL.Image
>>> import numpy as np


>>> def display_sample(sample, i):
...     image_processed = sample.cpu().permute(0, 2, 3, 1)
...     image_processed = (image_processed + 1.0) * 127.5
...     image_processed = image_processed.numpy().astype(np.uint8)

...     image_pil = PIL.Image.fromarray(image_processed[0])
...     display(f"Image at step {i}")
...     display(image_pil)
```

ノイズ除去処理を高速化するために入力とモデルをGPUに移します：

```py
>>> model.to("cuda")
>>> noisy_sample = noisy_sample.to("cuda")
```

ここで、ノイズが少なくなったサンプルの残りのノイズを予測するノイズ除去ループを作成し、スケジューラを使ってさらにノイズの少ないサンプルを計算します：

```py
>>> import tqdm

>>> sample = noisy_sample

>>> for i, t in enumerate(tqdm.tqdm(scheduler.timesteps)):
...     # 1. predict noise residual
...     with torch.no_grad():
...         residual = model(sample, t).sample

...     # 2. compute less noisy image and set x_t -> x_t-1
...     sample = scheduler.step(residual, t, sample).prev_sample

...     # 3. optionally look at image
...     if (i + 1) % 50 == 0:
...         display_sample(sample, i + 1)
```

何もないところから猫が生成されるのを、座って見てください！😻

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/diffusion-quicktour.png"/>
</div>

## 次のステップ

このクイックツアーで、🧨ディフューザーを使ったクールな画像をいくつか作成できたと思います！次のステップとして

* モデルをトレーニングまたは微調整については、[training](./tutorials/basic_training)チュートリアルを参照してください。
* 様々な使用例については、公式およびコミュニティの[training or finetuning scripts](https://github.com/huggingface/diffusers/tree/main/examples#-diffusers-examples)の例を参照してください。
* スケジューラのロード、アクセス、変更、比較については[Using different Schedulers](./using-diffusers/schedulers)ガイドを参照してください。
* プロンプトエンジニアリング、スピードとメモリの最適化、より高品質な画像を生成するためのヒントやトリックについては、[Stable Diffusion](./stable_diffusion)ガイドを参照してください。
* 🧨 Diffusers の高速化については、最適化された [PyTorch on a GPU](./optimization/fp16)のガイド、[Stable Diffusion on Apple Silicon (M1/M2)](./optimization/mps)と[ONNX Runtime](./optimization/onnx)を参照してください。
