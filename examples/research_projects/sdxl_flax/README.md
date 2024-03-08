# Stable Diffusion XL for JAX + TPUv5e

[TPU v5e](https://cloud.google.com/blog/products/compute/how-cloud-tpu-v5e-accelerates-large-scale-ai-inference) is a new generation of TPUs from Google Cloud. It is the most cost-effective, versatile, and scalable Cloud TPU to date. This makes them ideal for serving and scaling large diffusion models.

[JAX](https://github.com/google/jax) is a high-performance numerical computation library that is well-suited to develop and deploy diffusion models:

- **High performance**. All JAX operations are implemented in terms of operations in [XLA](https://www.tensorflow.org/xla/) - the Accelerated Linear Algebra compiler

- **Compilation**. JAX uses just-in-time (jit) compilation of JAX Python functions so it can be executed efficiently in XLA. In order to get the best performance, we must use static shapes for jitted functions, this is because JAX transforms work by tracing a function and to determine its effect on inputs of a specific shape and type. When a new shape is introduced to an already compiled function, it retriggers compilation on the new shape, which can greatly reduce performance. **Note**: JIT compilation is particularly well-suited for text-to-image generation because all inputs and outputs (image input / output sizes) are static.

- **Parallelization**. Workloads can be scaled across multiple devices using JAX's [pmap](https://jax.readthedocs.io/en/latest/_autosummary/jax.pmap.html), which expresses single-program multiple-data (SPMD) programs. Applying pmap to a function will compile a function with XLA, then execute in parallel on XLA devices. For text-to-image generation workloads this means that increasing the number of images rendered simultaneously is straightforward to implement and doesn't compromise performance.

👉 Try it out for yourself:

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/google/sdxl)

## Stable Diffusion XL pipeline in JAX

Upon having access to a TPU VM (TPUs higher than version 3), you should first install
a TPU-compatible version of JAX:
```
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

Next, we can install [flax](https://github.com/google/flax) and the diffusers library:

```
pip install flax diffusers transformers
```

In [sdxl_single.py](./sdxl_single.py) we give a simple example of how to write a text-to-image generation pipeline in JAX using [StabilityAI's Stable Diffusion XL](stabilityai/stable-diffusion-xl-base-1.0).

Let's explain it step-by-step:

**Imports and Setup**

```python
import jax
import jax.numpy as jnp
import numpy as np
from flax.jax_utils import replicate
from diffusers import FlaxStableDiffusionXLPipeline

from jax.experimental.compilation_cache import compilation_cache as cc
cc.initialize_cache("/tmp/sdxl_cache")
import time

NUM_DEVICES = jax.device_count()
```

First, we import the necessary libraries:
- `jax` is provides the primitives for TPU operations
- `flax.jax_utils` contains some useful utility functions for `Flax`, a neural network library built on top of JAX
- `diffusers` has all the code that is relevant for SDXL.
- We also initialize a cache to speed up the JAX model compilation.
- We automatically determine the number of available TPU devices.

**1. Downloading Model and Loading Pipeline**

```python
pipeline, params = FlaxStableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", revision="refs/pr/95", split_head_dim=True
)
```
Here, a pre-trained model `stable-diffusion-xl-base-1.0` from the namespace `stabilityai` is loaded. It returns a pipeline for inference and its parameters.

**2. Casting Parameter Types**

```python
scheduler_state = params.pop("scheduler")
params = jax.tree_util.tree_map(lambda x: x.astype(jnp.bfloat16), params)
params["scheduler"] = scheduler_state
```
This section adjusts the data types of the model parameters.
We convert all parameters to `bfloat16` to speed-up the computation with model weights.
**Note** that the scheduler parameters are **not** converted to `blfoat16` as the loss
in precision is degrading the pipeline's performance too significantly.

**3. Define Inputs to Pipeline**

```python
default_prompt = ...
default_neg_prompt = ...
default_seed = 33
default_guidance_scale = 5.0
default_num_steps = 25
```
Here, various default inputs for the pipeline are set, including the prompt, negative prompt, random seed, guidance scale, and the number of inference steps.

**4. Tokenizing Inputs**

```python
def tokenize_prompt(prompt, neg_prompt):
    prompt_ids = pipeline.prepare_inputs(prompt)
    neg_prompt_ids = pipeline.prepare_inputs(neg_prompt)
    return prompt_ids, neg_prompt_ids
```
This function tokenizes the given prompts. It's essential because the text encoders of SDXL don't understand raw text; they work with numbers. Tokenization converts text to numbers.

**5. Parallelization and Replication**

```python
p_params = replicate(params)

def replicate_all(prompt_ids, neg_prompt_ids, seed):
    ...
```
To utilize JAX's parallel capabilities, the parameters and input tensors are duplicated across devices. The `replicate_all` function also ensures that every device produces a different image by creating a unique random seed for each device.

**6. Putting Everything Together**

```python
def generate(...):
    ...
```
This function integrates all the steps to produce the desired outputs from the model. It takes in prompts, tokenizes them, replicates them across devices, runs them through the pipeline, and converts the images to a format that's more interpretable (PIL format).

**7. Compilation Step**

```python
start = time.time()
print(f"Compiling ...")
generate(default_prompt, default_neg_prompt)
print(f"Compiled in {time.time() - start}")
```
The initial run of the `generate` function will be slow because JAX compiles the function during this call. By running it once here, subsequent calls will be much faster. This section measures and prints the compilation time.

**8. Fast Inference**

```python
start = time.time()
prompt = ...
neg_prompt = ...
images = generate(prompt, neg_prompt)
print(f"Inference in {time.time() - start}")
```
Now that the function is compiled, this section shows how to use it for fast inference. It measures and prints the inference time.

In summary, the code demonstrates how to load a pre-trained model using Flax and JAX, prepare it for inference, and run it efficiently using JAX's capabilities.

## Ahead of Time (AOT) Compilation

FlaxStableDiffusionXLPipeline takes care of parallelization across multiple devices using jit. Now let's build parallelization ourselves.

For this we will be using a JAX feature called [Ahead of Time](https://jax.readthedocs.io/en/latest/aot.html) (AOT) lowering and compilation. AOT allows to fully compile prior to execution time and have control over different parts of the compilation process.

In [sdxl_single_aot.py](./sdxl_single_aot.py) we give a simple example of how to write our own parallelization logic for text-to-image generation pipeline in JAX using [StabilityAI's Stable Diffusion XL](stabilityai/stable-diffusion-xl-base-1.0)

We add a `aot_compile` function that compiles the `pipeline._generate` function
telling JAX which input arguments are static, that is, arguments that
are known at compile time and won't change. In our case, it is num_inference_steps,
height, width and return_latents.

Once the function is compiled, these parameters are omitted from future calls and
cannot be changed without modifying the code and recompiling.

```python
def aot_compile(
        prompt=default_prompt,
        negative_prompt=default_neg_prompt,
        seed=default_seed,
        guidance_scale=default_guidance_scale,
        num_inference_steps=default_num_steps
):
    prompt_ids, neg_prompt_ids = tokenize_prompt(prompt, negative_prompt)
    prompt_ids, neg_prompt_ids, rng = replicate_all(prompt_ids, neg_prompt_ids, seed)
    g = jnp.array([guidance_scale] * prompt_ids.shape[0], dtype=jnp.float32)
    g = g[:, None]

    return pmap(
        pipeline._generate,static_broadcasted_argnums=[3, 4, 5, 9]
        ).lower(
            prompt_ids,
            p_params,
            rng,
            num_inference_steps, # num_inference_steps
            height, # height
            width, # width
            g,
            None,
            neg_prompt_ids,
            False # return_latents
            ).compile()
````

Next we can compile the generate function by executing `aot_compile`.

```python
start = time.time()
print("Compiling ...")
p_generate = aot_compile()
print(f"Compiled in {time.time() - start}")
```
And again we put everything together in a `generate` function.

```python
def generate(
    prompt,
    negative_prompt,
    seed=default_seed,
    guidance_scale=default_guidance_scale
):
    prompt_ids, neg_prompt_ids = tokenize_prompt(prompt, negative_prompt)
    prompt_ids, neg_prompt_ids, rng = replicate_all(prompt_ids, neg_prompt_ids, seed)
    g = jnp.array([guidance_scale] * prompt_ids.shape[0], dtype=jnp.float32)
    g = g[:, None]
    images = p_generate(
        prompt_ids,
        p_params,
        rng,
        g,
        None,
        neg_prompt_ids)

    # convert the images to PIL
    images = images.reshape((images.shape[0] * images.shape[1], ) + images.shape[-3:])
    return pipeline.numpy_to_pil(np.array(images))
```

The first forward pass after AOT compilation still takes a while longer than
subsequent passes, this is because on the first pass, JAX uses Python dispatch, which
Fills the C++ dispatch cache.
When using jit, this extra step is done automatically, but when using AOT compilation,
it doesn't happen until the function call is made.

```python
start = time.time()
prompt = "photo of a rhino dressed suit and tie sitting at a table in a bar with a bar stools, award winning photography, Elke vogelsang"
neg_prompt = "cartoon, illustration, animation. face. male, female"
images = generate(prompt, neg_prompt)
print(f"First inference in {time.time() - start}")
```

From this point forward, any calls to generate should result in a faster inference
time and it won't change.

```python
start = time.time()
prompt = "photo of a rhino dressed suit and tie sitting at a table in a bar with a bar stools, award winning photography, Elke vogelsang"
neg_prompt = "cartoon, illustration, animation. face. male, female"
images = generate(prompt, neg_prompt)
print(f"Inference in {time.time() - start}")
```
