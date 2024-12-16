<!-- Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License. -->

# LTX

[LTX Video](https://huggingface.co/Lightricks/LTX-Video) is the first DiT-based video generation model capable of generating high-quality videos in real-time. It produces 24 FPS videos at a 768x512 resolution faster than they can be watched. Trained on a large-scale dataset of diverse videos, the model generates high-resolution videos with realistic and varied content. We provide a model for both text-to-video as well as image + text-to-video usecases.

<Tip>

Make sure to check out the Schedulers [guide](../../using-diffusers/schedulers.md) to learn how to explore the tradeoff between scheduler speed and quality, and see the [reuse components across pipelines](../../using-diffusers/loading.md#reuse-a-pipeline) section to learn how to efficiently load the same components into multiple pipelines.

</Tip>

## Loading Single Files

Loading the original LTX Video checkpoints is also possible with [`~ModelMixin.from_single_file`].

```python
import torch
from diffusers import AutoencoderKLLTXVideo, LTXImageToVideoPipeline, LTXVideoTransformer3DModel

single_file_url = "https://huggingface.co/Lightricks/LTX-Video/ltx-video-2b-v0.9.safetensors"
transformer = LTXVideoTransformer3DModel.from_single_file(
  single_file_url, torch_dtype=torch.bfloat16
)
vae = AutoencoderKLLTXVideo.from_single_file(single_file_url, torch_dtype=torch.bfloat16)
pipe = LTXImageToVideoPipeline.from_pretrained(
  "Lightricks/LTX-Video", transformer=transformer, vae=vae, torch_dtype=torch.bfloat16
)

# ... inference code ...
```

Alternatively, the pipeline can be used to load the weights with [`~FromSingleFileMixin.from_single_file`].

```python
import torch
from diffusers import LTXImageToVideoPipeline
from transformers import T5EncoderModel, T5Tokenizer

single_file_url = "https://huggingface.co/Lightricks/LTX-Video/ltx-video-2b-v0.9.safetensors"
text_encoder = T5EncoderModel.from_pretrained(
  "Lightricks/LTX-Video", subfolder="text_encoder", torch_dtype=torch.bfloat16
)
tokenizer = T5Tokenizer.from_pretrained(
  "Lightricks/LTX-Video", subfolder="tokenizer", torch_dtype=torch.bfloat16
)
pipe = LTXImageToVideoPipeline.from_single_file(
  single_file_url, text_encoder=text_encoder, tokenizer=tokenizer, torch_dtype=torch.bfloat16
)
```

Refer to [this section](https://huggingface.co/docs/diffusers/main/en/api/pipelines/cogvideox#memory-optimization) to learn more about optimizing memory consumption.

## LTXPipeline

[[autodoc]] LTXPipeline
  - all
  - __call__

## LTXImageToVideoPipeline

[[autodoc]] LTXImageToVideoPipeline
  - all
  - __call__

## LTXPipelineOutput

[[autodoc]] pipelines.ltx.pipeline_output.LTXPipelineOutput
