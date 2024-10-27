<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Inputs

Some model inputs are subclasses of [`~utils.BaseInput`], data structures containing all the information needed by the model. The inputs can also be used as tuples or dictionaries.

For example:

```python
from diffusers.models.controlnet_union import ControlNetUnionInput

union_input = ControlNetUnionInput(
    openpose=...
)
```

When considering the `inputs` object as a tuple, it considers all the attributes including those that have `None` values.

<Tip>

To check a specific pipeline or model input, refer to its corresponding API documentation.

</Tip>

## BaseInput

[[autodoc]] utils.BaseInput
    - to_tuple
