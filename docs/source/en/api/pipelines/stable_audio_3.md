<!--Copyright 2025 Stability AI and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Stable Audio 3

Stable Audio 3 (SA3) is a text-to-audio generation model from [Stability AI](https://stability.ai/). It uses a
distilled rectified-flow DiT with a ping-pong sampling schedule, enabling high-quality stereo audio at 44.1 kHz in
as few as 8 inference steps.

SA3 conditions on two signals:

* **Text** — encoded by a frozen T5Gemma encoder and injected via cross-attention.
* **Duration** — a single float (seconds) embedded by [`StableAudio3DurationEmbedder`] and broadcast to each DiT
  block as a global conditioning vector for adaptive layer normalisation.

The model uses a SAME (Semantically-Aligned Music Encoder) autoencoder ([`AutoencoderSAME`]) and the
[`PingPongScheduler`] for the default 8-step stochastic denoising loop.

The original codebase can be found at [Stability-AI/stable-audio-3](https://github.com/Stability-AI/stable-audio-3).

## Tips

* SA3 Medium is **adversarially distilled** — classifier-free guidance is baked into the weights. Do not pass a
  `negative_prompt`; there is no `guidance_scale` argument.
* The model is distilled for **exactly 8 ping-pong steps** (`num_inference_steps=8`). Other schedulers (e.g.
  [`FlowMatchEulerDiscreteScheduler`]) can be swapped in for experimentation.
* Pass `duration` in **seconds** (e.g. `duration=10.0`). The latent length is computed automatically.
* `silence_padding_duration` (default `0.0`) adds silent headroom at the end of the latent sequence. The reference
  implementation uses `6.0 s`; increase this value if you notice boundary artefacts.
* Multiple waveforms can be generated per prompt by setting `num_waveforms_per_prompt > 1`.

## StableAudio3Pipeline

[[autodoc]] StableAudio3Pipeline
	- all
	- __call__

## StableAudio3InpaintPipeline

[[autodoc]] StableAudio3InpaintPipeline
	- all
	- __call__

## StableAudio3DurationEmbedder

[[autodoc]] StableAudio3DurationEmbedder
