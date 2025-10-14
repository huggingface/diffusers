<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# LuminaFlowMatchScheduler

`LuminaFlowMatchScheduler` is a rectified flow scheduler designed for [Lumina-T2I](https://arxiv.org/abs/2405.05945). It implements flow matching that learns velocity fields to transport samples from noise to data distribution along straight paths.

## Overview

Rectified flow is a method for training and sampling from diffusion models that uses linear interpolation paths:

```
x_t = (1 - t) * noise + t * x_0
```

where the model learns to predict the velocity `v = x_0 - noise`.

The scheduler supports:

- Time shifting for better sampling quality
- Dynamic shifting based on image resolution
- Efficient Euler-based integration

This scheduler is specifically designed for the Lumina-T2I model but can be used with other flow-matching based models.

## LuminaFlowMatchScheduler

[[autodoc]] LuminaFlowMatchScheduler
