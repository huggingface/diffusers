<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# 概要

🧨 Diffusers には生成タスクのための多くのパイプライン、モデル、スケジューラがあります。 これらのコンポーネントをできる限りシンプルに読み込むために、私たちは単一で統一されたメソッドである - `from_pretrained()` - を提供します。 このメソッドは、Hugging Face [Hub](https://huggingface.co/models?library=diffusers&sort=downloads) またはローカルマシンから、これらのコンポーネントのいずれかを読み込みます。パイプラインやモデルを読み込むたびに、最新のファイルが自動的にダウンロードされキャッシュされることで、ファイルを再ダウンロードすることなく、すぐに再利用することができます。

このセクションでは、パイプラインの読み込み、パイプライン中の様々なコンポーネントの読み込み方、チェックポイントバリアントの読み込み方、コミュニティパイプラインの読み込み方について知っておくべき事を全て説明します。また、スケジューラを読み込む方法を学び、異なるスケジューラを使用した際の速度と品質のトレードオフを比較します。最後に、KerasCV チェックポイントを 🧨 Diffusers を使った PyTorch 上で扱うために、それらの変換方法や読み込み方を学びます。