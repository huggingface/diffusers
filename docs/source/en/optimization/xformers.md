<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Installing xFormers

We recommend the use of [xFormers](https://github.com/facebookresearch/xformers) for both inference and training. In our tests, the optimizations performed in the attention blocks allow for both faster speed and reduced memory consumption.

Starting from version `0.0.16` of xFormers, released on January 2023, installation can be easily performed using pre-built pip wheels:

```bash
pip install xformers
```

<Tip>

The xFormers PIP package requires the latest version of PyTorch (1.13.1 as of xFormers 0.0.16). If you need to use a previous version of PyTorch, then we recommend you install xFormers from source using [the project instructions](https://github.com/facebookresearch/xformers#installing-xformers).

</Tip>

After xFormers is installed, you can use `enable_xformers_memory_efficient_attention()` for faster inference and reduced memory consumption, as discussed [here](fp16#memory-efficient-attention).

<Tip warning={true}>

According to [this issue](https://github.com/huggingface/diffusers/issues/2234#issuecomment-1416931212), xFormers `v0.0.16` cannot be used for training (fine-tune or Dreambooth) in some GPUs. If you observe that problem, please install a development version as indicated in that comment.

</Tip>
