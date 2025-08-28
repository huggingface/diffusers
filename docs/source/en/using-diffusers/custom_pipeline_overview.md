<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

[[open-in-colab]]

# Community pipelines and components

Community pipelines are [`DiffusionPipeline`] classes that are different from the original paper implementation. They provide additional functionality or extend the original pipeline implementation.

> [!TIP]
> Check out the community pipelines in [diffusers/examples/community](https://github.com/huggingface/diffusers/tree/main/examples/community) with inference and training examples for how to use them.

Community pipelines are either stored on the Hub or the Diffusers' GitHub repository. Hub pipelines are completely customizable (scheduler, models, pipeline code, etc.) while GitHub pipelines are limited to only the custom pipeline code. Further compare the two community pipeline types in the table below.

|  | GitHub | Hub |
|---|---|---|
| Usage | Same. | Same. |
| Review process | Open a Pull Request on GitHub and undergo a review process from the Diffusers team before merging. This option is slower. | Upload directly to a Hub repository without a review. This is the fastest option. |
| Visibility | Included in the official Diffusers repository and docs. | Included on your Hub profile and relies on your own usage and promotion to gain visibility. |

## custom_pipeline

Load either community pipeline types by passing the `custom_pipeline` argument to [`~DiffusionPipeline.from_pretrained`].

```py
import torch
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers",
    custom_pipeline="pipeline_stable_diffusion_3_instruct_pix2pix",
    torch_dtype=torch.float16,
    device_map="cuda"
)
```

Add the `custom_revision` argument to [`~DiffusionPipeline.from_pretrained`] to load a community pipeline from a specific version (for example, `v0.30.0` or `main`). By default, community pipelines are loaded from the latest stable version of Diffusers.

```py
import torch
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers",
    custom_pipeline="pipeline_stable_diffusion_3_instruct_pix2pix",
    custom_revision="main"
    torch_dtype=torch.float16,
    device_map="cuda"
)
```

> [!WARNING]
> While the Hugging Face Hub [scans](https://huggingface.co/docs/hub/security-malware) files, you should still inspect the Hub pipeline code and make sure it is safe.

There are a few ways to load a community pipeline.

- Pass a path to `custom_pipeline` to load a local community pipeline. The directory must contain a `pipeline.py` file containing the pipeline class.

  ```py
  import torch
  from diffusers import DiffusionPipeline

  pipeline = DiffusionPipeline.from_pretrained(
      "stabilityai/stable-diffusion-3-medium-diffusers",
      custom_pipeline="path/to/pipeline_directory",
      torch_dtype=torch.float16,
      device_map="cuda"
  )
  ```

- The `custom_pipeline` argument is also supported by [`~DiffusionPipeline.from_pipe`], which is useful for [reusing pipelines](./loading#reuse-a-pipeline) without using additional memory. It limits the memory usage to only the largest pipeline loaded.

  ```py
  import torch
  from diffusers import DiffusionPipeline

  pipeline_sd = DiffusionPipeline.from_pretrained("emilianJR/CyberRealistic_V3", torch_dtype=torch.float16, device_map="cuda")
  pipeline_lpw = DiffusionPipeline.from_pipe(
      pipeline_sd, custom_pipeline="lpw_stable_diffusion", device_map="cuda"
  )
  ```

  The [`~DiffusionPipeline.from_pipe`] method is especially useful for loading community pipelines because many of them don't have pretrained weights. Community pipelines generally add a feature on top of an existing pipeline.

## Community components

Community components let users build pipelines with custom transformers, UNets, VAEs, and schedulers not supported by Diffusers. These components require Python module implementations. 

This section shows how users can use community components to build a community pipeline using [showlab/show-1-base](https://huggingface.co/showlab/show-1-base) as an example.

1. Load the required components, the scheduler and image processor. The text encoder is generally imported from [Transformers](https://huggingface.co/docs/transformers/index).

```python
from transformers import T5Tokenizer, T5EncoderModel, CLIPImageProcessor
from diffusers import DPMSolverMultistepScheduler

pipeline_id = "showlab/show-1-base"
tokenizer = T5Tokenizer.from_pretrained(pipeline_id, subfolder="tokenizer")
text_encoder = T5EncoderModel.from_pretrained(pipeline_id, subfolder="text_encoder")
scheduler = DPMSolverMultistepScheduler.from_pretrained(pipe_id, subfolder="scheduler")
feature_extractor = CLIPImageProcessor.from_pretrained(pipe_id, subfolder="feature_extractor")
```

> [!WARNING]
> In steps 2 and 3, the custom [UNet](https://github.com/showlab/Show-1/blob/main/showone/models/unet_3d_condition.py) and [pipeline](https://huggingface.co/sayakpaul/show-1-base-with-code/blob/main/unet/showone_unet_3d_condition.py) implementation must match the format shown in their files for this example to work.

2. Load a [custom UNet](https://github.com/showlab/Show-1/blob/main/showone/models/unet_3d_condition.py) which is already implemented in [showone_unet_3d_condition.py](https://huggingface.co/sayakpaul/show-1-base-with-code/blob/main/unet/showone_unet_3d_condition.py). The [`UNet3DConditionModel`] class name is renamed to the custom implementation, `ShowOneUNet3DConditionModel`, because [`UNet3DConditionModel`] already exists in Diffusers. Any components required for `ShowOneUNet3DConditionModel` class should be placed in `showone_unet_3d_condition.py`.

```python
from showone_unet_3d_condition import ShowOneUNet3DConditionModel

unet = ShowOneUNet3DConditionModel.from_pretrained(pipeline_id, subfolder="unet")
```

3. Load the custom pipeline code (already implemented in [pipeline_t2v_base_pixel.py](https://huggingface.co/sayakpaul/show-1-base-with-code/blob/main/pipeline_t2v_base_pixel.py)). This script contains a custom `TextToVideoIFPipeline` class for generating videos from text. Like the custom UNet, any code required for `TextToVideIFPipeline` should be placed in `pipeline_t2v_base_pixel.py`.

Initialize `TextToVideoIFPipeline` with `ShowOneUNet3DConditionModel`.

```python
import torch
from pipeline_t2v_base_pixel import TextToVideoIFPipeline

pipeline = TextToVideoIFPipeline(
    unet=unet,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    scheduler=scheduler,
    feature_extractor=feature_extractor,
    device_map="cuda",
    torch_dtype=torch.float16
)
```

4. Push the pipeline to the Hub to share with the community.

```python
pipeline.push_to_hub("custom-t2v-pipeline")
```

After the pipeline is successfully pushed, make the following changes.

- Change the `_class_name` attribute in [model_index.json](https://huggingface.co/sayakpaul/show-1-base-with-code/blob/main/model_index.json#L2) to `"pipeline_t2v_base_pixel"` and `"TextToVideoIFPipeline"`.
- Upload `showone_unet_3d_condition.py` to the [unet](https://huggingface.co/sayakpaul/show-1-base-with-code/blob/main/unet/showone_unet_3d_condition.py) subfolder.
- Upload `pipeline_t2v_base_pixel.py` to the pipeline [repository](https://huggingface.co/sayakpaul/show-1-base-with-code/tree/main).

To run inference, add the `trust_remote_code` argument while initializing the pipeline to handle all the "magic" behind the scenes.

```python
import torch
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
    "<change-username>/<change-id>", trust_remote_code=True, torch_dtype=torch.float16
)
```

> [!WARNING]
> As an additional precaution with `trust_remote_code=True`, we strongly encourage passing a commit hash to the `revision` argument in [`~DiffusionPipeline.from_pretrained`] to make sure the code hasn't been updated with new malicious code (unless you fully trust the model owners).

## Resources

- Take a look at Issue [#841](https://github.com/huggingface/diffusers/issues/841) for more context about why we're adding community pipelines to help everyone easily share their work without being slowed down.
- Check out the [stabilityai/japanese-stable-diffusion-xl](https://huggingface.co/stabilityai/japanese-stable-diffusion-xl/) repository for an additional example of a community pipeline that also uses the `trust_remote_code` feature.