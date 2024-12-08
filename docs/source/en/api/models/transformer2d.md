<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Transformer2DModel

A Transformer model for image-like data from [CompVis](https://huggingface.co/CompVis) that is based on the [Vision Transformer](https://huggingface.co/papers/2010.11929) introduced by Dosovitskiy et al. The [`Transformer2DModel`] accepts discrete (classes of vector embeddings) or continuous (actual embeddings) inputs.

When the input is **continuous**:

1. Project the input and reshape it to `(batch_size, sequence_length, feature_dimension)`.
2. Apply the Transformer blocks in the standard way.
3. Reshape to image.

When the input is **discrete**:

<Tip>

It is assumed one of the input classes is the masked latent pixel. The predicted classes of the unnoised image don't contain a prediction for the masked pixel because the unnoised image cannot be masked.

</Tip>

1. Convert input (classes of latent pixels) to embeddings and apply positional embeddings.
2. Apply the Transformer blocks in the standard way.
3. Predict classes of unnoised image.

## Transformer2DModel

[[autodoc]] Transformer2DModel

## Transformer2DModelOutput

[[autodoc]] models.modeling_outputs.Transformer2DModelOutput
