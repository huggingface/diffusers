<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Load safetensors

[[open-in-colab]]

[safetensors](https://github.com/huggingface/safetensors) is a safe and fast file format for storing and loading tensors. Typically, PyTorch model weights are saved or *pickled* into a `.bin` file with Python's [`pickle`](https://docs.python.org/3/library/pickle.html) utility. However, `pickle` is not secure and pickled files may contain malicious code that can be executed. safetensors is a secure alternative to `pickle`, making it ideal for sharing model weights.

This guide will show you how you load `.safetensor` files, and how to convert Stable Diffusion model weights stored in other formats to `.safetensor`. Before you start, make sure you have safetensors installed:

```py
# uncomment to install the necessary libraries in Colab
#!pip install safetensors
```

If you look at the [`runwayml/stable-diffusion-v1-5`](https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main) repository, you'll see weights inside the `text_encoder`, `unet` and `vae` subfolders are stored in the `.safetensors` format. By default, 🤗 Diffusers automatically loads these `.safetensors` files from their subfolders if they're available in the model repository.

For more explicit control, you can optionally set `use_safetensors=True` (if `safetensors` is not installed, you'll get an error message asking you to install it):

```py
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", use_safetensors=True)
```

However, model weights are not necessarily stored in separate subfolders like in the example above. Sometimes, all the weights are stored in a single `.safetensors` file. In this case, if the weights are Stable Diffusion weights, you can load the file directly with the [`~diffusers.loaders.FromSingleFileMixin.from_single_file`] method:

```py
from diffusers import StableDiffusionPipeline

pipeline = StableDiffusionPipeline.from_single_file(
    "https://huggingface.co/WarriorMama777/OrangeMixs/blob/main/Models/AbyssOrangeMix/AbyssOrangeMix.safetensors"
)
```

## Convert to safetensors

Not all weights on the Hub are available in the `.safetensors` format, and you may encounter weights stored as `.bin`. In this case, use the [Convert Space](https://huggingface.co/spaces/diffusers/convert) to convert the weights to `.safetensors`. The Convert Space downloads the pickled weights, converts them, and opens a Pull Request to upload the newly converted `.safetensors` file on the Hub. This way, if there is any malicious code contained in the pickled files, they're uploaded to the Hub - which has a [security scanner](https://huggingface.co/docs/hub/security-pickle#hubs-security-scanner) to detect unsafe files and suspicious pickle imports - instead of your computer.

You can use the model with the new `.safetensors` weights by specifying the reference to the Pull Request in the `revision` parameter (you can also test it in this [Check PR](https://huggingface.co/spaces/diffusers/check_pr) Space on the Hub), for example `refs/pr/22`:

```py
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1", revision="refs/pr/22", use_safetensors=True
)
```

## Why use safetensors?

There are several reasons for using safetensors:

- Safety is the number one reason for using safetensors. As open-source and model distribution grows, it is important to be able to trust the model weights you downloaded don't contain any malicious code. The current size of the header in safetensors prevents parsing extremely large JSON files.
- Loading speed between switching models is another reason to use safetensors, which performs zero-copy of the tensors. It is especially fast compared to `pickle` if you're loading the weights to CPU (the default case), and just as fast if not faster when directly loading the weights to GPU. You'll only notice the performance difference if the model is already loaded, and not if you're downloading the weights or loading the model for the first time.

	The time it takes to load the entire pipeline:

	```py
 	from diffusers import StableDiffusionPipeline

 	pipeline = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", use_safetensors=True)
 	"Loaded in safetensors 0:00:02.033658"
 	"Loaded in PyTorch 0:00:02.663379"
	```

	But the actual time it takes to load 500MB of the model weights is only:

	```bash
	safetensors: 3.4873ms
	PyTorch: 172.7537ms
	```

- Lazy loading is also supported in safetensors, which is useful in distributed settings to only load some of the tensors. This format allowed the [BLOOM](https://huggingface.co/bigscience/bloom) model to be loaded in 45 seconds on 8 GPUs instead of 10 minutes with regular PyTorch weights.
