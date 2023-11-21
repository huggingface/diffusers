<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Textual Inversion

Textual Inversion is a training method for personalizing models by learning new text embeddings from a few example images. The file produced from training is extremely small (a few KBs) and the new embeddings can be loaded into the text encoder.

[`TextualInversionLoaderMixin`] provides a function for loading Textual Inversion embeddings from Diffusers and Automatic1111 into the text encoder and loading a special token to activate the embeddings.

<Tip>

To learn more about how to load Textual Inversion embeddings, see the [Textual Inversion](../../using-diffusers/loading_adapters#textual-inversion) loading guide.

</Tip>

## TextualInversionLoaderMixin

[[autodoc]] loaders.textual_inversion.TextualInversionLoaderMixin