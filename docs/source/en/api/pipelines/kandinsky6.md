<!--Copyright 2026 The Kandinsky Team and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Kandinsky 6

Kandinsky 6 provides pipelines for text/image-to-video-and-audio generation
and video super-resolution.

[`Kandinsky6TI2VAPipeline`] denoises video and audio latents together with a single multimodal transformer, conditioned on a Qwen2.5-VL text encoder (and, optionally, a reference image) plus a CLIP text encoder for pooled embeddings. Video and audio come out of the same denoising loop, so there is no separate vocoder or post-hoc audio pass; pass `sample_audio=False` to skip generating audio and drop the `audio_vae` component.

[`Kandinsky6SRPipeline`] upscales the video latents produced by the base pipeline. Its production route (`resolution_scale=2.25`) does a 1.125x pixel pre-upscale followed by an x2 latent-upscale step, using [`Kandinsky6SRLatentUpscalerBank`] as the latent upscaler; a `source_vae` is only needed for the KVAE latent bridge and is otherwise unused.

## Kandinsky6TI2VAPipeline

[[autodoc]] Kandinsky6TI2VAPipeline
  - all
  - __call__

## Kandinsky6SRPipeline

[[autodoc]] Kandinsky6SRPipeline
  - all
  - __call__

## Kandinsky6SRLatentUpscalerBank

[[autodoc]] Kandinsky6SRLatentUpscalerBank
  - all
  - forward

## Pipeline outputs

[[autodoc]] Kandinsky6TI2VAPipelineOutput

[[autodoc]] Kandinsky6SRPipelineOutput
