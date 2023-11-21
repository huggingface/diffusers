<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# DPMSolverSinglestepScheduler

`DPMSolverSinglestepScheduler` is a single step scheduler from [DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps](https://huggingface.co/papers/2206.00927) and [DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models](https://huggingface.co/papers/2211.01095) by Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu.

DPMSolver (and the improved version DPMSolver++) is a fast dedicated high-order solver for diffusion ODEs with convergence order guarantee. Empirically, DPMSolver sampling with only 20 steps can generate high-quality
samples, and it can generate quite good samples even in 10 steps.

The original implementation can be found at [LuChengTHU/dpm-solver](https://github.com/LuChengTHU/dpm-solver).

## Tips

It is recommended to set `solver_order` to 2 for guide sampling, and `solver_order=3` for unconditional sampling.

Dynamic thresholding from [Imagen](https://huggingface.co/papers/2205.11487) is supported, and for pixel-space
diffusion models, you can set both `algorithm_type="dpmsolver++"` and `thresholding=True` to use dynamic
thresholding. This thresholding method is unsuitable for latent-space diffusion models such as
Stable Diffusion.

## DPMSolverSinglestepScheduler
[[autodoc]] DPMSolverSinglestepScheduler

## SchedulerOutput
[[autodoc]] schedulers.scheduling_utils.SchedulerOutput
