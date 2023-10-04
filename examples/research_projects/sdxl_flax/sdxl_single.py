# Show best practices for SDXL JAX
import time

import jax
import jax.numpy as jnp
import numpy as np
from flax.jax_utils import replicate

# Let's cache the model compilation, so that it doesn't take as long the next time around.
from jax.experimental.compilation_cache import compilation_cache as cc

from diffusers import FlaxStableDiffusionXLPipeline


cc.initialize_cache("/tmp/sdxl_cache")


NUM_DEVICES = jax.device_count()

# 1. Let's start by downloading the model and loading it into our pipeline class
# Adhering to JAX's functional approach, the model's parameters are returned seperatetely and
# will have to be passed to the pipeline during inference
pipeline, params = FlaxStableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", revision="refs/pr/95", split_head_dim=True
)

# 2. We cast all parameters to bfloat16 EXCEPT the scheduler which we leave in
# float32 to keep maximal precision
scheduler_state = params.pop("scheduler")
params = jax.tree_util.tree_map(lambda x: x.astype(jnp.bfloat16), params)
params["scheduler"] = scheduler_state

# 3. Next, we define the different inputs to the pipeline
default_prompt = "a colorful photo of a castle in the middle of a forest with trees and bushes, by Ismail Inceoglu, shadows, high contrast, dynamic shading, hdr, detailed vegetation, digital painting, digital drawing, detailed painting, a detailed digital painting, gothic art, featured on deviantart"
default_neg_prompt = "fog, grainy, purple"
default_seed = 33
default_guidance_scale = 5.0
default_num_steps = 25


# 4. In order to be able to compile the pipeline
# all inputs have to be tensors or strings
# Let's tokenize the prompt and negative prompt
def tokenize_prompt(prompt, neg_prompt):
    prompt_ids = pipeline.prepare_inputs(prompt)
    neg_prompt_ids = pipeline.prepare_inputs(neg_prompt)
    return prompt_ids, neg_prompt_ids


# 5. To make full use of JAX's parallelization capabilities
# the parameters and input tensors are duplicated across devices
# To make sure every device generates a different image, we create
# different seeds for each image. The model parameters won't change
# during inference so we do not wrap them into a function
p_params = replicate(params)


def replicate_all(prompt_ids, neg_prompt_ids, seed):
    p_prompt_ids = replicate(prompt_ids)
    p_neg_prompt_ids = replicate(neg_prompt_ids)
    rng = jax.random.PRNGKey(seed)
    rng = jax.random.split(rng, NUM_DEVICES)
    return p_prompt_ids, p_neg_prompt_ids, rng


# 6. Let's now put it all together in a generate function
def generate(
    prompt,
    negative_prompt,
    seed=default_seed,
    guidance_scale=default_guidance_scale,
    num_inference_steps=default_num_steps,
):
    prompt_ids, neg_prompt_ids = tokenize_prompt(prompt, negative_prompt)
    prompt_ids, neg_prompt_ids, rng = replicate_all(prompt_ids, neg_prompt_ids, seed)
    images = pipeline(
        prompt_ids,
        p_params,
        rng,
        num_inference_steps=num_inference_steps,
        neg_prompt_ids=neg_prompt_ids,
        guidance_scale=guidance_scale,
        jit=True,
    ).images

    # convert the images to PIL
    images = images.reshape((images.shape[0] * images.shape[1],) + images.shape[-3:])
    return pipeline.numpy_to_pil(np.array(images))


# 7. Remember that the first call will compile the function and hence be very slow. Let's run generate once
# so that the pipeline call is compiled
start = time.time()
print("Compiling ...")
generate(default_prompt, default_neg_prompt)
print(f"Compiled in {time.time() - start}")

# 8. Now the model forward pass will run very quickly, let's try it again
start = time.time()
prompt = "photo of a rhino dressed suit and tie sitting at a table in a bar with a bar stools, award winning photography, Elke vogelsang"
neg_prompt = "cartoon, illustration, animation. face. male, female"
images = generate(prompt, neg_prompt)
print(f"Inference in {time.time() - start}")

for i, image in enumerate(images):
    image.save(f"castle_{i}.png")
