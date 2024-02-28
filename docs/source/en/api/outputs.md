<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Outputs

All model outputs are subclasses of [`~utils.BaseOutput`], data structures containing all the information returned by the model. The outputs can also be used as tuples or dictionaries.

For example:

```python
from diffusers import DDIMPipeline

pipeline = DDIMPipeline.from_pretrained("google/ddpm-cifar10-32")
outputs = pipeline()
```

The `outputs` object is a [`~pipelines.ImagePipelineOutput`] which means it has an image attribute.

You can access each attribute as you normally would or with a keyword lookup, and if that attribute is not returned by the model, you will get `None`:

```python
outputs.images
outputs["images"]
```

When considering the `outputs` object as a tuple, it only considers the attributes that don't have `None` values.
For instance, retrieving an image by indexing into it returns the tuple `(outputs.images)`:

```python
outputs[:1]
```

<Tip>

To check a specific pipeline or model output, refer to its corresponding API documentation.

</Tip>

## BaseOutput

[[autodoc]] utils.BaseOutput
    - to_tuple

## ImagePipelineOutput

[[autodoc]] pipelines.ImagePipelineOutput

## FlaxImagePipelineOutput

[[autodoc]] pipelines.pipeline_flax_utils.FlaxImagePipelineOutput

## AudioPipelineOutput

[[autodoc]] pipelines.AudioPipelineOutput

## ImageTextPipelineOutput

[[autodoc]] ImageTextPipelineOutput
