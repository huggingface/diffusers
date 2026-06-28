<!--Copyright 2025 Stability AI and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# AutoencoderSAME

The **SAME** (Semantically-Aligned Music Encoder) autoencoder is used by [Stable Audio 3](https://stability.ai/news/stable-audio-3)
to compress stereo audio waveforms into a compact latent sequence and reconstruct them.

The encoder stacks [`SAMETransformerResamplingBlock`] modules, each of which groups a fixed number of audio
patch frames and produces one learnable output token via a differential transformer. The decoder inverts this
process, expanding each latent token back to a patch of audio frames.

A soft-norm bottleneck ([`_SoftNormBottleneck`]) normalises latents before and after the diffusion model,
providing stable training dynamics.

## AutoencoderSAME

[[autodoc]] AutoencoderSAME
	- all
	- encode
	- decode

## SAMETransformerResamplingBlock

[[autodoc]] SAMETransformerResamplingBlock

## AutoencoderSAMEOutput

[[autodoc]] AutoencoderSAMEOutput

## AutoencoderSAMEDecoderOutput

[[autodoc]] AutoencoderSAMEDecoderOutput
