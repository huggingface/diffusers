<!-- Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License. -->

# Qwen-Image 2.1

Qwen-Image 2.1 encodes the prompt and any condition images together with a Qwen3-VL model, then denoises the target
image with a single-stream block-causal transformer. See
[`QwenImage21Transformer2DModel`](../models/qwenimage21_transformer2d) for what `causal_block` and `causal_condition`
change.

Because the text and condition-image prefix is modulated from `t = 0`, its keys and values do not change between
denoising steps. The pipeline caches them after the first step by default; pass `use_kv_cache=False` to recompute the
full sequence every step.

Toggling `use_kv_cache` does not reproduce the same image bit-for-bit in reduced precision. The cached decode step
attends with a different sequence layout than the prefill step, so the two land on different rounding — both match an
fp32 reference to the same tolerance — and a one-ULP difference at the first block is amplified by 32 blocks and every
sampler step. Keep the flag fixed when you need a reproducible sample.

<Tip>

This pipeline requires the `flex` attention backend when `causal_block` is enabled.

</Tip>

## QwenImage21Pipeline

[[autodoc]] QwenImage21Pipeline
    - all
    - __call__

## QwenImagePipelineOutput

[[autodoc]] pipelines.qwenimage.pipeline_output.QwenImagePipelineOutput
