<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# コミュニティのパイプラインとコンポーネントの読み込み

[[open-in-colab]]

## コミュニティパイプライン

コミュニティパイプラインは、任意の [`DiffusionPipeline`] クラスであり、論文で提案されたオリジナルの実装とは異なります。
(たとえば、 [`StableDiffusionControlNetPipeline`] は [Text-to-Image Generation with ControlNet Conditioning](https://arxiv.org/abs/2302.05543) に対応します。)
それらはパイプラインの元の実装を拡張したり、追加機能を提供したりします。

[Speech to Image](https://github.com/huggingface/diffusers/tree/main/examples/community#speech-to-image) や [Composable Stable Diffusion](https://github.com/huggingface/diffusers/tree/main/examples/community#composable-stable-diffusion) のような多くの素晴らしいコミュニティパイプラインがあります。
すべての公式コミュニティパイプラインは[ここ](https://github.com/huggingface/diffusers/tree/main/examples/community)から確認することができます。

コミュニティパイプラインを Hub に読み込むためには、`custom_pipeline` 引数にコミュニティパイプラインのリポジトリ ID を渡し、
パイプラインの重みとコンポーネントを読み込むモデルリポジトリを指定します。
たとえば、以下の例では [`hf-internal-testing/diffusers-dummy-pipeline`](https://huggingface.co/hf-internal-testing/diffusers-dummy-pipeline/blob/main/pipeline.py) からダミーパイプラインを読み込み、
[`google/ddpm-cifar10-32`](https://huggingface.co/google/ddpm-cifar10-32) 


<Tip warning={true}>

🔒 Hugging Face Hub からコミニティパイプラインを読み込むことで、読み込むコードが安全であることを信頼する必要があります。実行する前に必ずオンラインでコードを確認してください！

</Tip>

```py
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
    "google/ddpm-cifar10-32", custom_pipeline="hf-internal-testing/diffusers-dummy-pipeline", use_safetensors=True
)
```

公式コミュニティのパイプラインを読み込むことも同様ですが、
公式リポジトリ ID から重みを読み込むことと、パイプラインのコンポーネントを直接渡すことは併用することができます。
以下の例では、コミュニティの [CLIP Guided Stable Diffusion](https://github.com/huggingface/diffusers/tree/main/examples/community#clip-guided-stable-diffusion) パイプラインを読み込み、
CLIP モデルのコンポーネントを直接渡すことができます:

```py
from diffusers import DiffusionPipeline
from transformers import CLIPImageProcessor, CLIPModel

clip_model_id = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"

feature_extractor = CLIPImageProcessor.from_pretrained(clip_model_id)
clip_model = CLIPModel.from_pretrained(clip_model_id)

pipeline = DiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    custom_pipeline="clip_guided_stable_diffusion",
    clip_model=clip_model,
    feature_extractor=feature_extractor,
    use_safetensors=True,
)
```

コミュニティパイプラインの詳細については、[コミュニティパイプライン](custom_pipeline_examples) ガイドを参照してください。
また、コミュティパイプラインの追加に興味がある場合は、[コミュニティパイプラインの追加方法](contribute_pipeline) ガイドを参照してください！

## コミュニティコンポーネント

コミュニティコンポーネントによって、ユーザは Diffusers にない、カスタマイズされたパイプラインを構築することができます。
もし、あなたのパイプラインが Diffusers がまだサポートしていないカスタムコンポーネントを持つ場合は、それらの実装を Python モジュールとして提供する必要があります。
これらのカスタマイズされたコンポーネントは、VAE、UNet、スケジューラがあります。
ほとんどの場合、テキストエンコーダは Transformers ライブラリからインポートされます。
パイプライン自体もカスタマイズ可能です。

このセクションでは、ユーザがコミュニティコンポーネントを使用して、コミュニティパイプラインを構築する方法を示します。

例として、[showlab/show-1-base](https://huggingface.co/showlab/show-1-base) パイプラインのチェックポイントを使います。
それでは、コンポーネントの読み込みを始めましょう:

1. Transformers からテキストエンコーダをインポートして読み込む:

```python
from transformers import T5Tokenizer, T5EncoderModel

pipe_id = "showlab/show-1-base"
tokenizer = T5Tokenizer.from_pretrained(pipe_id, subfolder="tokenizer")
text_encoder = T5EncoderModel.from_pretrained(pipe_id, subfolder="text_encoder")
```

2. スケジューラを読み込む:

```python
from diffusers import DPMSolverMultistepScheduler

scheduler = DPMSolverMultistepScheduler.from_pretrained(pipe_id, subfolder="scheduler")
```

3. 画像プロセッサを読み込む:

```python
from transformers import CLIPFeatureExtractor

feature_extractor = CLIPFeatureExtractor.from_pretrained(pipe_id, subfolder="feature_extractor")
```

<Tip warning={true}>

ステップ4と5において、カスタム [UNet](https://github.com/showlab/Show-1/blob/main/showone/models/unet_3d_condition.py) と[パイプライン](https://huggingface.co/sayakpaul/show-1-base-with-code/blob/main/unet/showone_unet_3d_condition.py)の実装は、この例が動作するために、それらのファイルに示されているフォーマットと一致していなければいけません。

</Tip>

4. この例では便宜上 `showone_unet_3d_condition.py` [スクリプト](https://huggingface.co/sayakpaul/show-1-base-with-code/blob/main/unet/showone_unet_3d_condition.py)で既に実装されている[カスタム UNet]((https://github.com/showlab/Show-1/blob/main/showone/models/unet_3d_condition.py)を読み込んでいきましょう。`UNet3DConditionModel` クラスが `ShowOneUNet3DConditionModel` クラスに変更されていることに気づくと思います。　`ShowOneUNet3DConditionModel` クラスに必要なコンポーネントは `showone_unet_3d_condition.py` スクリプトに配置してください。 

これが完了したら、UNetを初期化することができます:

```python
from showone_unet_3d_condition import ShowOneUNet3DConditionModel

unet = ShowOneUNet3DConditionModel.from_pretrained(pipe_id, subfolder="unet")
```

5. 最後に、カスタムパイプラインのコードを読み込みます。この例では、既に `pipeline_t2v_base_pixel.py`  [スクリプト](https://huggingface.co/sayakpaul/show-1-base-with-code/blob/main/pipeline_t2v_base_pixel.py)に作られています。この、スクリプトにはテキストから動画を生成するためのカスタムクラス `TextToVideoIFPipeline` が含まれています。カスタム Unet のように、カスタムパイプラインに必要なコードは `pipeline_t2v_base_pixel.py` スクリプトに記述してください。

すべての準備が整ったら、`TextToVideoIFPipeline` を `ShowOneUNet3DConditionModel` で初期化します:

```python
from pipeline_t2v_base_pixel import TextToVideoIFPipeline
import torch

pipeline = TextToVideoIFPipeline(
    unet=unet,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    scheduler=scheduler,
    feature_extractor=feature_extractor
)
pipeline = pipeline.to(device="cuda")
pipeline.torch_dtype = torch.float16
```

パイプラインを Hub にプッシュして、コミュニティと共有しましょう！

```python
pipeline.push_to_hub("custom-t2v-pipeline")
```

パイプラインのプッシュが成功したら、いくつかの変更が必要です:

1. [`model_index.json`](https://huggingface.co/sayakpaul/show-1-base-with-code/blob/main/model_index.json#L2) の `_class_name` 属性を `"pipeline_t2v_base_pixel"` と `"TextToVideoIFPipeline"` に変更
2. `showone_unet_3d_condition.py` を `unet` [ディレクトリ](https://huggingface.co/sayakpaul/show-1-base-with-code/blob/main/unet/showone_unet_3d_condition.py) にアップロード
3. `pipeline_t2v_base_pixel.py` を パイプラインベース[ディレクトリ](https://huggingface.co/sayakpaul/show-1-base-with-code/blob/main/unet/showone_unet_3d_condition.py)にアップロード

推論を実行するには、パイプラインの初期化時に `trust_remote_code` 引数を追加するだけで、舞台裏で全ての「マジック」を処理することができます。

```python
from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained(
    "<change-username>/<change-id>", trust_remote_code=True, torch_dtype=torch.float16
).to("cuda")

prompt = "hello"

# テキスト埋め込み
prompt_embeds, negative_embeds = pipeline.encode_prompt(prompt)

# キーフレームの生成 (8x64x40, 2fps)
video_frames = pipeline(
    prompt_embeds=prompt_embeds,
    negative_prompt_embeds=negative_embeds,
    num_frames=8,
    height=40,
    width=64,
    num_inference_steps=2,
    guidance_scale=9.0,
    output_type="pt"
).frames
```

さらなる参考例として、`trust_remote_code` を利用した [stabilityai/japanese-stable-diffusion-xl](https://huggingface.co/stabilityai/japanese-stable-diffusion-xl/) のリポジトリ構造を参照することができます。

```python

from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained(
    "stabilityai/japanese-stable-diffusion-xl", trust_remote_code=True
)
pipeline.to("cuda")

# if using torch < 2.0
# pipeline.enable_xformers_memory_efficient_attention()

prompt = "柴犬、カラフルアート"

image = pipeline(prompt=prompt).images[0]

```