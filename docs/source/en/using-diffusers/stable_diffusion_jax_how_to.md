<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# JAX/Flax

[[open-in-colab]]

🤗 Diffusers supports Flax for super fast inference on Google TPUs, such as those available in Colab, Kaggle or Google Cloud Platform. This guide shows you how to run inference with Stable Diffusion using JAX/Flax.

Before you begin, make sure you have the necessary libraries installed:

```py
# uncomment to install the necessary libraries in Colab
#!pip install -q jax==0.3.25 jaxlib==0.3.25 flax transformers ftfy
#!pip install -q diffusers
```

You should also make sure you're using a TPU backend. While JAX does not run exclusively on TPUs, you'll get the best performance on a TPU because each server has 8 TPU accelerators working in parallel.

If you are running this guide in Colab, select *Runtime* in the menu above, select the option *Change runtime type*, and then select *TPU* under the *Hardware accelerator* setting. Import JAX and quickly check whether you're using a TPU:

```python
import jax
import jax.tools.colab_tpu
jax.tools.colab_tpu.setup_tpu()

num_devices = jax.device_count()
device_type = jax.devices()[0].device_kind

print(f"Found {num_devices} JAX devices of type {device_type}.")
assert (
    "TPU" in device_type, 
    "Available device is not a TPU, please select TPU from Edit > Notebook settings > Hardware accelerator"
)
"Found 8 JAX devices of type Cloud TPU."
```

Great, now you can import the rest of the dependencies you'll need:

```python
import numpy as np
import jax.numpy as jnp

from pathlib import Path
from jax import pmap
from flax.jax_utils import replicate
from flax.training.common_utils import shard
from PIL import Image

from huggingface_hub import notebook_login
from diffusers import FlaxStableDiffusionPipeline
```

## Load a model

Flax is a functional framework, so models are stateless and parameters are stored outside of them. Loading a pretrained Flax pipeline returns *both* the pipeline and the model weights (or parameters). In this guide, you'll use `bfloat16`, a more efficient half-float type that is supported by TPUs (you can also use `float32` for full precision if you want).

```python
dtype = jnp.bfloat16
pipeline, params = FlaxStableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    revision="bf16",
    dtype=dtype,
)
```

## Inference

TPUs usually have 8 devices working in parallel, so let's use the same prompt for each device. This means you can perform inference on 8 devices at once, with each device generating one image. As a result, you'll get 8 images in the same amount of time it takes for one chip to generate a single image!

<Tip>

Learn more details in the [How does parallelization work?](#how-does-parallelization-work) section.

</Tip>

After replicating the prompt, get the tokenized text ids by calling the `prepare_inputs` function on the pipeline. The length of the tokenized text is set to 77 tokens as required by the configuration of the underlying CLIP text model.

```python
prompt = "A cinematic film still of Morgan Freeman starring as Jimi Hendrix, portrait, 40mm lens, shallow depth of field, close up, split lighting, cinematic"
prompt = [prompt] * jax.device_count()
prompt_ids = pipeline.prepare_inputs(prompt)
prompt_ids.shape
"(8, 77)"
```

Model parameters and inputs have to be replicated across the 8 parallel devices. The parameters dictionary is replicated with [`flax.jax_utils.replicate`](https://flax.readthedocs.io/en/latest/api_reference/flax.jax_utils.html#flax.jax_utils.replicate) which traverses the dictionary and changes the shape of the weights so they are repeated 8 times. Arrays are replicated using `shard`.

```python
# parameters
p_params = replicate(params)

# arrays
prompt_ids = shard(prompt_ids)
prompt_ids.shape
"(8, 1, 77)"
```

This shape means each one of the 8 devices receives as an input a `jnp` array with shape `(1, 77)`, where `1` is the batch size per device. On TPUs with sufficient memory, you could have a batch size larger than `1` if you want to generate multiple images (per chip) at once.

Next, create a random number generator to pass to the generation function. This is standard procedure in Flax, which is very serious and opinionated about random numbers. All functions that deal with random numbers are expected to receive a generator to ensure reproducibility, even when you're training across multiple distributed devices.

The helper function below uses a seed to initialize a random number generator. As long as you use the same seed, you'll get the exact same results. Feel free to use different seeds when exploring results later in the guide.

```python
def create_key(seed=0):
    return jax.random.PRNGKey(seed)
```

The helper function, or `rng`, is split 8 times so each device receives a different generator and generates a different image.

```python
rng = create_key(0)
rng = jax.random.split(rng, jax.device_count())
```

To take advantage of JAX's optimized speed on a TPU, pass `jit=True` to the pipeline to compile the JAX code into an efficient representation and to ensure the model runs in parallel across the 8 devices.

<Tip warning={true}>

You need to ensure all your inputs have the same shape in subsequent calls, other JAX will need to recompile the code which is slower.

</Tip>

The first inference run takes more time because it needs to compile the code, but subsequent calls (even with different inputs) are much faster. For example, it took more than a minute to compile on a TPU v2-8, but then it takes about **7s** on a future inference run!

```py
%%time
images = pipeline(prompt_ids, p_params, rng, jit=True)[0]

"CPU times: user 56.2 s, sys: 42.5 s, total: 1min 38s"
"Wall time: 1min 29s"
```

The returned array has shape `(8, 1, 512, 512, 3)` which should be reshaped to remove the second dimension and get 8 images of `512 × 512 × 3`. Then you can use the [`~utils.numpy_to_pil`] function to convert the arrays into images.

```python
from diffusers import make_image_grid

images = images.reshape((images.shape[0] * images.shape[1],) + images.shape[-3:])
images = pipeline.numpy_to_pil(images)
make_image_grid(images, 2, 4)
```

![img](https://huggingface.co/datasets/YiYiXu/test-doc-assets/resolve/main/stable_diffusion_jax_how_to_cell_38_output_0.jpeg)

## Using different prompts

You don't necessarily have to use the same prompt on all devices. For example, to generate 8 different prompts:

```python
prompts = [
    "Labrador in the style of Hokusai",
    "Painting of a squirrel skating in New York",
    "HAL-9000 in the style of Van Gogh",
    "Times Square under water, with fish and a dolphin swimming around",
    "Ancient Roman fresco showing a man working on his laptop",
    "Close-up photograph of young black woman against urban background, high quality, bokeh",
    "Armchair in the shape of an avocado",
    "Clown astronaut in space, with Earth in the background",
]

prompt_ids = pipeline.prepare_inputs(prompts)
prompt_ids = shard(prompt_ids)

images = pipeline(prompt_ids, p_params, rng, jit=True).images
images = images.reshape((images.shape[0] * images.shape[1],) + images.shape[-3:])
images = pipeline.numpy_to_pil(images)

make_image_grid(images, 2, 4)
```

![img](https://huggingface.co/datasets/YiYiXu/test-doc-assets/resolve/main/stable_diffusion_jax_how_to_cell_43_output_0.jpeg)


## How does parallelization work?

The Flax pipeline in 🤗 Diffusers automatically compiles the model and runs it in parallel on all available devices. Let's take a closer look at how that process works.

JAX parallelization can be done in multiple ways. The easiest one revolves around using the [`jax.pmap`](https://jax.readthedocs.io/en/latest/_autosummary/jax.pmap.html) function to achieve single-program multiple-data (SPMD) parallelization. It means running several copies of the same code, each on different data inputs. More sophisticated approaches are possible, and you can go over to the JAX [documentation](https://jax.readthedocs.io/en/latest/index.html) to explore this topic in more detail if you are interested!

`jax.pmap` does two things:

1. Compiles (or "`jit`s") the code which is similar to `jax.jit()`. This does not happen when you call `pmap`, and only the first time the `pmap`ped function is called.
2. Ensures the compiled code runs in parallel on all available devices.

To demonstrate, call `pmap` on the pipeline's `_generate` method (this is a private method that generates images and may be renamed or removed in future releases of 🤗 Diffusers):

```python
p_generate = pmap(pipeline._generate)
```

After calling `pmap`, the prepared function `p_generate` will:

1. Make a copy of the underlying function, `pipeline._generate`, on each device.
2. Send each device a different portion of the input arguments (this is why its necessary to call the *shard* function). In this case, `prompt_ids` has shape `(8, 1, 77, 768)` so the array is split into 8 and each copy of `_generate` receives an input with shape `(1, 77, 768)`.

The most important thing to pay attention to here is the batch size (1 in this example), and the input dimensions that make sense for your code. You don't have to change anything else to make the code work in parallel.

The first time you call the pipeline takes more time, but the calls afterward are much faster. The `block_until_ready` function is used to correctly measure inference time because JAX uses asynchronous dispatch and returns control to the Python loop as soon as it can. You don't need to use that in your code; blocking occurs automatically when you want to use the result of a computation that has not yet been materialized.

```py
%%time
images = p_generate(prompt_ids, p_params, rng)
images = images.block_until_ready()
"CPU times: user 1min 15s, sys: 18.2 s, total: 1min 34s"
"Wall time: 1min 15s"
```

Check your image dimensions to see if they're correct:

```python
images.shape
"(8, 1, 512, 512, 3)"
```