<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# 개요

노이즈가 많은 출력에서 적은 출력으로 만드는 과정으로 고품질 생성 모델의 출력을 만드는 각각의 반복되는 스텝은 많은 계산이 필요합니다. 🧨 Diffuser의 목표 중 하나는 모든 사람이 이 기술을 널리 이용할 수 있도록 하는 것이며, 여기에는 소비자 및 특수 하드웨어에서 빠른 추론을 가능하게 하는 것을 포함합니다.

이 섹션에서는 추론 속도를 최적화하고 메모리 소비를 줄이기 위한 반정밀(half-precision) 가중치 및 sliced attention과 같은 팁과 요령을 다룹니다. 또한 [`torch.compile`](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) 또는 [ONNX Runtime](https://onnxruntime.ai/docs/)을 사용하여 PyTorch 코드의 속도를 높이고, [xFormers](https://facebookresearch.github.io/xformers/)를 사용하여 memory-efficient attention을 활성화하는 방법을 배울 수 있습니다. Apple Silicon, Intel 또는 Habana 프로세서와 같은 특정 하드웨어에서 추론을 실행하기 위한 가이드도 있습니다.