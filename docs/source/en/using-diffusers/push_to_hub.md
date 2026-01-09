<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

[[open-in-colab]]

# Sharing pipelines and models

Share your pipeline or models and schedulers on the Hub with the [`~diffusers.utils.PushToHubMixin`] class. This class:

1. creates a repository on the Hub
2. saves your model, scheduler, or pipeline files so they can be reloaded later
3. uploads folder containing these files to the Hub

This guide will show you how to upload your files to the Hub with the [`~diffusers.utils.PushToHubMixin`] class.

Log in to your Hugging Face account with your access [token](https://huggingface.co/settings/tokens).

<hfoptions id="login">
<hfoption id="notebook">

```py
from huggingface_hub import notebook_login

notebook_login()
```

</hfoption>
<hfoption id="hf CLI">

```bash
hf auth login
```

</hfoption>
</hfoptions>

## Models

To push a model to the Hub, call [`~diffusers.utils.PushToHubMixin.push_to_hub`] and specify the repository id of the model.

```py
from diffusers import ControlNetModel

controlnet = ControlNetModel(
    block_out_channels=(32, 64),
    layers_per_block=2,
    in_channels=4,
    down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
    cross_attention_dim=32,
    conditioning_embedding_out_channels=(16, 32),
)
controlnet.push_to_hub("my-controlnet-model")
```

The [`~diffusers.utils.PushToHubMixin.push_to_hub`] method saves the model's `config.json` file and the weights are automatically saved as safetensors files.

Load the model again with [`~DiffusionPipeline.from_pretrained`].

```py
model = ControlNetModel.from_pretrained("your-namespace/my-controlnet-model")
```

## Scheduler

To push a scheduler to the Hub, call [`~diffusers.utils.PushToHubMixin.push_to_hub`] and specify the repository id of the scheduler.

```py
from diffusers import DDIMScheduler

scheduler = DDIMScheduler(
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
)
scheduler.push_to_hub("my-controlnet-scheduler")
```

The [`~diffusers.utils.PushToHubMixin.push_to_hub`] function saves the scheduler's `scheduler_config.json` file to the specified repository.

Load the scheduler again with [`~SchedulerMixin.from_pretrained`].

```py
scheduler = DDIMScheduler.from_pretrained("your-namepsace/my-controlnet-scheduler")
```

## Pipeline

To push a pipeline to the Hub, initialize the pipeline components with your desired parameters.

```py
from diffusers import (
    UNet2DConditionModel,
    AutoencoderKL,
    DDIMScheduler,
    StableDiffusionPipeline,
)
from transformers import CLIPTextModel, CLIPTextConfig, CLIPTokenizer

unet = UNet2DConditionModel(
    block_out_channels=(32, 64),
    layers_per_block=2,
    sample_size=32,
    in_channels=4,
    out_channels=4,
    down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
    up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
    cross_attention_dim=32,
)

scheduler = DDIMScheduler(
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
)

vae = AutoencoderKL(
    block_out_channels=[32, 64],
    in_channels=3,
    out_channels=3,
    down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
    up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
    latent_channels=4,
)

text_encoder_config = CLIPTextConfig(
    bos_token_id=0,
    eos_token_id=2,
    hidden_size=32,
    intermediate_size=37,
    layer_norm_eps=1e-05,
    num_attention_heads=4,
    num_hidden_layers=5,
    pad_token_id=1,
    vocab_size=1000,
)
text_encoder = CLIPTextModel(text_encoder_config)
tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")
```

Pass all components to the pipeline and call [`~diffusers.utils.PushToHubMixin.push_to_hub`].

```py
components = {
    "unet": unet,
    "scheduler": scheduler,
    "vae": vae,
    "text_encoder": text_encoder,
    "tokenizer": tokenizer,
    "safety_checker": None,
    "feature_extractor": None,
}

pipeline = StableDiffusionPipeline(**components)
pipeline.push_to_hub("my-pipeline")
```

The [`~diffusers.utils.PushToHubMixin.push_to_hub`] method saves each component to a subfolder in the repository. Load the pipeline again with [`~DiffusionPipeline.from_pretrained`].

```py
pipeline = StableDiffusionPipeline.from_pretrained("your-namespace/my-pipeline")
```

## Privacy

Set `private=True` in [`~diffusers.utils.PushToHubMixin.push_to_hub`] to keep a model, scheduler, or pipeline files private.

```py
controlnet.push_to_hub("my-controlnet-model-private", private=True)
```

Private repositories are only visible to you. Other users won't be able to clone the repository and it won't appear in search results. Even if a user has the URL to your private repository, they'll receive a `404 - Sorry, we can't find the page you are looking for`. You must be [logged in](https://huggingface.co/docs/huggingface_hub/quick-start#login) to load a model from a private repository.