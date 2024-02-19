<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

<p align="center">
    <br>
    <img src="https://raw.githubusercontent.com/huggingface/diffusers/77aadfee6a891ab9fcfb780f87c693f7a5beeb8e/docs/source/imgs/diffusers_library.jpg" width="400"/>
    <br>
</p>

# Diffusers

🤗 Diffusers は、画像や音声、さらには分子の3D構造を生成するための、最先端の事前学習済みDiffusion Model(拡散モデル)を提供するライブラリです。シンプルな生成ソリューションをお探しの場合でも、独自の拡散モデルをトレーニングしたい場合でも、🤗 Diffusers はその両方をサポートするモジュール式のツールボックスです。私たちのライブラリは、[性能より使いやすさ](conceptual/philosophy#usability-over-performance)、[簡単よりシンプル](conceptual/philosophy#simple-over-easy)、[抽象化よりカスタマイズ性](conceptual/philosophy#tweakable-contributorfriendly-over-abstraction)に重点を置いて設計されています。

このライブラリには3つの主要コンポーネントがあります:

- 数行のコードで推論可能な最先端の[拡散パイプライン](api/pipelines/overview)。Diffusersには多くのパイプラインがあります。利用可能なパイプラインを網羅したリストと、それらが解決するタスクについては、パイプラインの[概要](https://huggingface.co/docs/diffusers/api/pipelines/overview)の表をご覧ください。
- 生成速度と品質のトレードオフのバランスを取る交換可能な[ノイズスケジューラ](api/schedulers/overview)
- ビルディングブロックとして使用することができ、スケジューラと組み合わせることで、エンドツーエンドの拡散モデルを構築可能な事前学習済み[モデル](api/models)

<div class="mt-10">
  <div class="w-full flex flex-col space-y-4 md:space-y-0 md:grid md:grid-cols-2 md:gap-y-4 md:gap-x-5">
    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg" href="./tutorials/tutorial_overview"
      ><div class="w-full text-center bg-gradient-to-br from-blue-400 to-blue-500 rounded-lg py-1.5 font-semibold mb-5 text-white text-lg leading-relaxed">チュートリアル</div>
      <p class="text-gray-700">出力の生成、独自の拡散システムの構築、拡散モデルのトレーニングを開始するために必要な基本的なスキルを学ぶことができます。初めて 🤗Diffusersを使用する場合は、ここから始めることをおすすめします！</p>
    </a>
    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg" href="./using-diffusers/loading_overview"
      ><div class="w-full text-center bg-gradient-to-br from-indigo-400 to-indigo-500 rounded-lg py-1.5 font-semibold mb-5 text-white text-lg leading-relaxed">ガイド</div>
      <p class="text-gray-700">パイプライン、モデル、スケジューラの読み込みに役立つ実践的なガイドです。また、特定のタスクにパイプラインを使用する方法、出力の生成方法を制御する方法、生成速度を最適化する方法、さまざまなトレーニング手法についても学ぶことができます。</p>
    </a>
    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg" href="./conceptual/philosophy"
      ><div class="w-full text-center bg-gradient-to-br from-pink-400 to-pink-500 rounded-lg py-1.5 font-semibold mb-5 text-white text-lg leading-relaxed">Conceptual guides</div>
      <p class="text-gray-700">ライブラリがなぜこのように設計されたのかを理解し、ライブラリを利用する際の倫理的ガイドラインや安全対策について詳しく学べます。</p>
   </a>
    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg" href="./api/models/overview"
      ><div class="w-full text-center bg-gradient-to-br from-purple-400 to-purple-500 rounded-lg py-1.5 font-semibold mb-5 text-white text-lg leading-relaxed">リファレンス</div>
      <p class="text-gray-700">🤗 Diffusersのクラスとメソッドがどのように機能するかについての技術的な説明です。</p>
    </a>
  </div>
</div>