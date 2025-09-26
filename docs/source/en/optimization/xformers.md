<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# xFormers

We recommend [xFormers](https://github.com/facebookresearch/xformers) for both inference and training. In our tests, the optimizations performed in the attention blocks allow for both faster speed and reduced memory consumption.

Install xFormers from `pip`:

```bash
pip install xformers
```

> [!TIP]
> The xFormers `pip` package requires the latest version of PyTorch. If you need to use a previous version of PyTorch, then we recommend [installing xFormers from the source](https://github.com/facebookresearch/xformers#installing-xformers).

After xFormers is installed, you can use `enable_xformers_memory_efficient_attention()` for faster inference and reduced memory consumption as shown in this [section](memory#memory-efficient-attention).

> [!WARNING]
> According to this [issue](https://github.com/huggingface/diffusers/issues/2234#issuecomment-1416931212), xFormers `v0.0.16` cannot be used for training (fine-tune or DreamBooth) in some GPUs. If you observe this problem, please install a development version as indicated in the issue comments.
