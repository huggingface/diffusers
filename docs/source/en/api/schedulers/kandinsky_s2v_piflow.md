<!--Copyright 2026 The Kandinsky Team and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# KandinskyS2VPiflowScheduler

`KandinskyS2VPiflowScheduler` is the few-step pi-Flow scheduler of the distilled Kandinsky S2V 1.0
checkpoints. The student predicts a grid of x0 values per segment in one forward
pass, and the scheduler integrates that policy across the segment, so each
segment costs a single model evaluation.

## KandinskyS2VPiflowScheduler
[[autodoc]] KandinskyS2VPiflowScheduler
