<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License. -->

# AutoencoderKLCogVideoX

The 3D variational autoencoder (VAE) model with KL loss using CogVideoX.

## Loading from the original format

By default, the [`AutoencoderKLCogVideoX`] should be loaded with [`~ModelMixin.from_pretrained`], but it can also be loaded from the original format using [`FromOriginalModelMixin.from_single_file`] as follows:

```py
from diffusers import AutoencoderKLCogVideoX

url = "THUDM/CogVideoX-2b"  # can also be a local file
model = AutoencoderKLCogVideoX.from_single_file(url)

```

## AutoencoderKLCogVideoX

[[autodoc]] AutoencoderKLCogVideoX
    - decode
    - encode
    - all

## CogVideoXSafeConv3d

[[autodoc]] CogVideoXSafeConv3d

## CogVideoXCausalConv3d

[[autodoc]] CogVideoXCausalConv3d

## CogVideoXSpatialNorm3D

[[autodoc]] CogVideoXSpatialNorm3D

## CogVideoXResnetBlock3D

[[autodoc]] CogVideoXResnetBlock3D

## CogVideoXDownBlock3D

[[autodoc]] CogVideoXDownBlock3D

## CogVideoXMidBlock3D

[[autodoc]] CogVideoXMidBlock3D

## CogVideoXUpBlock3D

[[autodoc]] CogVideoXUpBlock3D

## CogVideoXEncoder3D

[[autodoc]] CogVideoXEncoder3D

## CogVideoXDecoder3D

[[autodoc]] CogVideoXDecoder3D