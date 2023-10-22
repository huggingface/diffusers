<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# インストール

Install 🤗 Diffusers for whichever deep learning library you're working with.

🤗 Diffusers is tested on Python 3.8+, PyTorch 1.7.0+ and Flax. Follow the installation instructions below for the deep learning library you are using:

- [PyTorch](https://pytorch.org/get-started/locally/) installation instructions.
- [Flax](https://flax.readthedocs.io/en/latest/) installation instructions.

## Install with pip

You should install 🤗 Diffusers in a [virtual environment](https://docs.python.org/3/library/venv.html).
If you're unfamiliar with Python virtual environments, take a look at this [guide](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/).
A virtual environment makes it easier to manage different projects and avoid compatibility issues between dependencies.

Start by creating a virtual environment in your project directory:

```bash
python -m venv .env
```

Activate the virtual environment:

```bash
source .env/bin/activate
```

🤗 Diffusers also relies on the 🤗 Transformers library, and you can install both with the following command:

<frameworkcontent>
<pt>
```bash
pip install diffusers["torch"] transformers
```
</pt>
<jax>
```bash
pip install diffusers["flax"] transformers
```
</jax>
</frameworkcontent>

## Install from source

ソースからのインストール

ソースから🤗 Diffusersをインストールする前に、`torch`と🤗 Accelerateがインストールされていることを確認してください。

`torch`のインストールについては、`torch` [インストール](https://pytorch.org/get-started/locally/#start-locally)ガイドを参照してください。

🤗 Accelerateをインストールするには：

```bash
pip install accelerate
```

以下のコマンドでソースから🤗 Diffusersをインストールできます：

```bash
pip install git+https://github.com/huggingface/diffusers
```

このコマンドは最新の `stable` バージョンではなく、最先端の `main` バージョンをインストールします。
`main`バージョンは最新の開発に対応するのに便利です。
例えば、前回の公式リリース以降にバグが修正されたが、新しいリリースがまだリリースされていない場合などには都合がいいです。
しかし、これは `main` バージョンが常に安定しているとは限らないです。
私たちは `main` バージョンを運用し続けるよう努力しており、ほとんどの問題は通常数時間から1日以内に解決されます。
もし問題が発生した場合は、[Issue](https://github.com/huggingface/diffusers/issues/new/choose) を開いてください！

## 編集可能なインストール

以下の場合、編集可能なインストールが必要です：

* ソースコードの `main` バージョンを使用する。
* 🤗 Diffusers に貢献し、コードの変更をテストする必要がある場合。

リポジトリをクローンし、次のコマンドで 🤗 Diffusers をインストールしてください：

```bash
git clone https://github.com/huggingface/diffusers.git
cd diffusers
```

<frameworkcontent>
<pt>
```bash
pip install -e ".[torch]"
```
</pt>
<jax>
```bash
pip install -e ".[flax]"
```
</jax>
</frameworkcontent>

これらのコマンドは、リポジトリをクローンしたフォルダと Python のライブラリパスをリンクします。
Python は通常のライブラリパスに加えて、クローンしたフォルダの中を探すようになります。
例えば、Python パッケージが通常 `~/anaconda3/envs/main/lib/python3.8/site-packages/` にインストールされている場合、Python はクローンした `~/diffusers/` フォルダも同様に参照します。

<Tip warning={true}>

ライブラリを使い続けたい場合は、`diffusers`フォルダを残しておく必要があります。

</Tip>

これで、以下のコマンドで簡単にクローンを最新版の🤗 Diffusersにアップデートできます：

```bash
cd ~/diffusers/
git pull
```

Python環境は次の実行時に `main` バージョンの🤗 Diffusersを見つけます。

## テレメトリー・ロギングに関するお知らせ

このライブラリは `from_pretrained()` リクエスト中にデータを収集します。
このデータには Diffusers と PyTorch/Flax のバージョン、要求されたモデルやパイプラインクラスが含まれます。
また、Hubでホストされている場合は、事前に学習されたチェックポイントへのパスが含まれます。
この使用データは問題のデバッグや新機能の優先順位付けに役立ちます。
テレメトリーはHuggingFace Hubからモデルやパイプラインをロードするときのみ送信されます。ローカルでの使用中は収集されません。

我々は、すべての人が追加情報を共有したくないことを理解し、あなたのプライバシーを尊重します。
そのため、ターミナルから `DISABLE_TELEMETRY` 環境変数を設定することで、データ収集を無効にすることができます：

Linux/MacOSの場合
```bash
export DISABLE_TELEMETRY=YES
```

Windows の場合
```bash
set DISABLE_TELEMETRY=YES
```
