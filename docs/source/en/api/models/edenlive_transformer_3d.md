<!-- Copyright 2026 The VAST-AI Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# EdenLiveTransformer3DModel

The EdenLive diffusion transformer predicts flow velocities from video latents, text embeddings, and projected
DINO proxy features. The same class supports full-video bidirectional attention and causal attention over successive
chunks. The pipeline loads the configuration and weights for the selected profile; see the
[pipeline guide](../pipelines/edenlive) for original-checkpoint loading and video generation.

## Inputs and output

Call `forward()` with the following inputs. Here `B` is batch size, `T` is the number of latent frames in the video or
current chunk, and `H` and `W` are latent spatial dimensions.

| Argument | Shape or format | Meaning |
|---|---|---|
| `hidden_states` | `(B, in_channels, T, H, W)` | Noisy latents, plus image-conditioning channels for 14B |
| `timestep` | Scalar, `(B,)`, `(B, T)`, or `(B, num_patch_tokens)` | Model time; retain FP32 even with BF16 model weights |
| `encoder_hidden_states` | `(B, sequence_length, text_dim)` | UMT5 text embeddings |
| `encoder_hidden_states_image` | `(B, image_sequence_length, image_dim)` | CLIP reference-image embeddings, required for 14B and unused by 5B |
| `proxy_features` | `(B, proxy_channels, T, proxy_height, proxy_width)` | Required projected DINO features; spatial dimensions are interpolated to match the latents |

The temporal patch size is one; latent height and width must be divisible by the spatial patch size, which is two
for the released profiles. Proxy features must match the latent batch and frame counts. The 5B model uses 48 input
and output latent channels. The 14B model receives 36 input channels: 16 noisy latent channels, 16 image latent
channels, and four mask channels; it predicts 16 output channels. Proxy channels are appended internally and are
not included in `in_channels`.

With `return_dict=True`, `forward()` returns `Transformer2DModelOutput.sample` with shape
`(B, out_channels, T, H, W)`. With `return_dict=False`, it returns the same prediction as the first tuple entry.
The prediction is a latent flow velocity, not a decoded video.

The transformer's `text_timeline` is an internal token layout for 5B bidirectional inference. It contains `lengths`
(positive query-token counts per segment and sample) and `text_length` (the number of tokens per text prompt).
User-facing timelines in seconds are accepted and converted by `EdenLivePipeline`; they cannot be passed directly
to the transformer.

## Causal attention state

Use `create_causal_state(batch_size, height, width, ...)` to allocate state for a rollout, with height and width in
latent coordinates. The released defaults are three latent frames per chunk, a maximum context of 18 latent frames,
and six retained sink frames. Allocate a separate state for each classifier-free guidance branch.

Denoising calls pass `causal_state=state` and `cache_update=False`. After a chunk has been denoised, a clean call
at timestep zero with `cache_update=True` commits its keys and values and advances the frame cursor. A short chunk
is supported only at the end of the rollout. The pipeline manages these calls and releases its states after every
invocation. For custom loops, call `state.reset()` before beginning an independent rollout.

## EdenLiveTransformer3DModel API

[[autodoc]] EdenLiveTransformer3DModel
    - forward
    - create_causal_state

## EdenLiveProxyProjection

The trained projection maps the last channel dimension of DINO features to `out_channels`, applies L2 normalization
on that dimension, and returns FP32 features. Inputs can retain arbitrary leading batch, frame, or spatial token
dimensions. The pipeline reshapes the result into the channel-first video layout expected by the transformer.

Its `in_channels` must match the DINO encoder feature width and `out_channels` must match the transformer's
`proxy_channels`. Use the released trained projection together with its DINO encoder.

[[autodoc]] EdenLiveProxyProjection
    - forward

## EdenLiveCausalState

State belongs to one rollout and guidance branch and is not part of the model checkpoint. Create it through the
transformer so cache allocation matches the model's layers, dtype, device, and context-parallel layout.

| Field | Meaning |
|---|---|
| `layers` | Per-layer unrotated keys, values, and valid-token counts |
| `batch_size`, `height`, `width` | Batch and latent spatial dimensions used to allocate the caches |
| `chunk_size`, `max_context_frames`, `sink_frames` | Causal context limits in latent frames |
| `tokens_per_frame` | Spatial patch-token count per latent frame |
| `cp_size`, `cp_rank` | Context-parallel degree and local rank |
| `next_frame` | Number of latent frames committed so far; also the next chunk's position offset |

`reset()` clears valid-token counts and resets `next_frame` to zero, allowing the allocation to be reused for an
independent rollout with the same shape and parallel layout.

[[autodoc]] EdenLiveCausalState
    - reset
