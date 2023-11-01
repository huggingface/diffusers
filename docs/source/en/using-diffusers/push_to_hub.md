<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Push files to the Hub

[[open-in-colab]]

🤗 Diffusers provides a [`~diffusers.utils.PushToHubMixin`] for uploading your model, scheduler, or pipeline to the Hub. It is an easy way to store your files on the Hub, and also allows you to share your work with others. Under the hood, the [`~diffusers.utils.PushToHubMixin`]:

1. creates a repository on the Hub
2. saves your model, scheduler, or pipeline files so they can be reloaded later
3. uploads folder containing these files to the Hub

This guide will show you how to use the [`~diffusers.utils.PushToHubMixin`] to upload your files to the Hub.

You'll need to log in to your Hub account with your access [token](https://huggingface.co/settings/tokens) first:

```py
from huggingface_hub import notebook_login

notebook_login()
```

## Models

To push a model to the Hub, call [`~diffusers.utils.PushToHubMixin.push_to_hub`] and specify the repository id of the model to be stored on the Hub:

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

For models, you can also specify the [*variant*](loading#checkpoint-variants) of the weights to push to the Hub. For example, to push `fp16` weights:

```py
controlnet.push_to_hub("my-controlnet-model", variant="fp16")
```

The [`~diffusers.utils.PushToHubMixin.push_to_hub`] function saves the model's `config.json` file and the weights are automatically saved in the `safetensors` format.

Now you can reload the model from your repository on the Hub:

```py
model = ControlNetModel.from_pretrained("your-namespace/my-controlnet-model")
```

## Scheduler

To push a scheduler to the Hub, call [`~diffusers.utils.PushToHubMixin.push_to_hub`] and specify the repository id of the scheduler to be stored on the Hub:

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

Now you can reload the scheduler from your repository on the Hub:

```py
scheduler = DDIMScheduler.from_pretrained("your-namepsace/my-controlnet-scheduler")
```

## Pipeline

You can also push an entire pipeline with all it's components to the Hub. For example, initialize the components of a [`StableDiffusionPipeline`] with the parameters you want:

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

Pass all of the components to the [`StableDiffusionPipeline`] and call [`~diffusers.utils.PushToHubMixin.push_to_hub`] to push the pipeline to the Hub:

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

The [`~diffusers.utils.PushToHubMixin.push_to_hub`] function saves each component to a subfolder in the repository. Now you can reload the pipeline from your repository on the Hub:

```py
pipeline = StableDiffusionPipeline.from_pretrained("your-namespace/my-pipeline")
```

## Privacy

Set `private=True` in the [`~diffusers.utils.PushToHubMixin.push_to_hub`] function to keep your model, scheduler, or pipeline files private:

```py
controlnet.push_to_hub("my-controlnet-model-private", private=True)
```

Private repositories are only visible to you, and other users won't be able to clone the repository and your repository won't appear in search results. Even if a user has the URL to your private repository, they'll receive a `404 - Sorry, we can't find the page you are looking for.`

To load a model, scheduler, or pipeline from private or gated repositories, set `use_auth_token=True`:

```py
model = ControlNetModel.from_pretrained("your-namespace/my-controlnet-model-private", use_auth_token=True)
```
