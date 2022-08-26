# 🧨 Diffusers Pipelines

Pipelines provide a simple API to run diffusion systems that possibly include multiple different modeling 
compenents in inference.

**Note** Pipelines do and should not offer an any functionality to train independent diffusion models or other components of a diffusion system, such as text encoders, image generation models, or super-resolution models. If you are looking for *offical* training examples, please have a look at [exampels](https://github.com/huggingface/diffusers/tree/main/examples).

## How to diffusers pipelines work


## How to write your own diffusion pipeline

Diffusers pipelines consists of 
- Pipelines are a collection of end-to-end diffusion systems that can be used out-of-the-box
- Pipelines should stay as close as possible to their original implementation 
- Pipelines can include components of other library, such as text-encoders. 

### Philosophy

Our examples aspire to be **self-contained**, **easy-to-tweak**, **beginner-friendly** and for **one-purpose-only**.
More specifically, this means:

- **Self-contained**: An example script shall only depend on "pip-install-able" Python packages that can be found in a `requirements.txt` file. Example scripts shall **not** depend on any local files. This means that one can simply download an example script, *e.g.* [train_unconditional.py](https://github.com/huggingface/diffusers/blob/main/examples/unconditional_image_generation/train_unconditional.py), install the required dependencies, *e.g.* [requirements.txt](https://github.com/huggingface/diffusers/blob/main/examples/unconditional_image_generation/requirements) and execute the example script.
- **Easy-to-tweak**: While we strive to present as many use cases as possible, the example scripts are just that - examples. It is expected that they won't work out-of-the box on your specific problem and that you will be required to change a few lines of code to adapt them to your needs. To help you with that, most of the examples fully expose the preprocessing of the data and the training loop to allow you to tweak and edit them as required.
- **Beginner-friendly**: We do not aim for providing state-of-the-art training scripts for the newest models, but rather examples that can be used as a way to better understand Diffusion models and how to use them with the `diffusers` library. We often purposefully leave out certain state-of-the-art methods if we consider them too complex for beginners.
- **One-purpose-only**: Examples should show one task and one task only. Even if a task is from a modeling 
point of view very similar, *e.g.* image super-resolution and image modification tend to use the same model and training method, we want examples to showcase only one task to keep them as readable and easy-to-understand as possible.

### Examples

- DDPM for unconditional image generation in [pipeline_ddpm](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/ddpm/pipeline_ddpm.py).
- DDIM for unconditional image generation in [pipeline_ddim](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/ddim/pipeline_ddim.py).
- PNDM for unconditional image generation in [pipeline_pndm](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/pndm/pipeline_pndm.py).
- Latent diffusion for text to image generation / conditional image generation in [pipeline_latent_diffusion](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/latent_diffusion/pipeline_latent_diffusion.py).
- Glide for text to image generation / conditional image generation in [pipeline_glide](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/glide/pipeline_glide.py).
- BDDMPipeline for spectrogram-to-sound vocoding in [pipeline_bddm](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/bddm/pipeline_bddm.py).
- Grad-TTS for text to audio generation / conditional audio generation in [pipeline_grad_tts](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/grad_tts/pipeline_grad_tts.py).
