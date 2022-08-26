# 🧨 Diffusers Pipelines

Pipelines provide a simply way to run state-of-the-art difffusion models in inference.
Most diffusion models consits of multiple independently-trained models and highly adaptable scheduler 
components - all of which are needed to have a functioning end-to-end diffusion model.

As an example, [Stable Diffusion](https://huggingface.co/blog/stable_diffusion) has three indepently trained models:
- [Autoencoder](https://github.com/huggingface/diffusers/blob/5cbed8e0d157f65d3ddc2420dfd09f2df630e978/src/diffusers/models/vae.py#L392)
- [Conditional Unet](https://github.com/huggingface/diffusers/blob/5cbed8e0d157f65d3ddc2420dfd09f2df630e978/src/diffusers/models/unet_2d_condition.py#L12)
- [CLIP text encoder](https://huggingface.co/docs/transformers/v4.21.2/en/model_doc/clip#transformers.CLIPTextModel)
- a scheduler component, [scheduler](https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_pndm.py), 
- a [CLIPFeatureExtractor](https://huggingface.co/docs/transformers/v4.21.2/en/model_doc/clip#transformers.CLIPFeatureExtractor),
- as well as a [safety checker](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/safety_checker.py).
All of these components are necessary to run stable diffusion in inference even thought they were trained 
or created independently from each other.

To that end, we strive to offer all open-sourced, state-of-the-art diffusion models under a unified API. 
More specifically, we strive to provide pipelines that
- 1. can load the officially published weights and yield 1-to-1 the same outputs as the original implemetation according to the corresponding paper (*e.g.* [LatentDiffusionPipeline](https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines/latent_diffusion), uses the officially released weights of [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)),
- 2. have a simple user interface to run the model in inference (see the [Pipelines API](#pipelines-api) section), 
- 3. are easy to understand with code that is self-explanatory and be can read along-side the official paper (see [Pipelines summary](#pipelines-summary)),
- 4. can easily be contributed by the community (see the [Contribution](#contribution) section).

**Note** that pipelines do and should not offer any training functionality. 
If you are looking for *official* training examples, please have a look at [examples](https://github.com/huggingface/diffusers/tree/main/examples).


## Pipelines Summary

The following ta

<To be filled>

## Pipelines API

Diffusion models often consists of multiple independently trained models or created componets. 


Each model has been trained indepently on a different task and the scheduler can easily be swapped out against another scheduler. 
During inference, we however want to be able to easily load all components and use them in inference - even if one component, *e.g.* CLIP's text encoder, originates from a different library, such as [Transformers](https://github.com/huggingface/transformers). To that end, all pipelines provide the following functionality:

- [`from_pretrained` method](https://github.com/huggingface/diffusers/blob/5cbed8e0d157f65d3ddc2420dfd09f2df630e978/src/diffusers/pipeline_utils.py#L139) that accepts a Hugging Face Hub repository id, *e.g.* [CompVis/stable-diffusion-v1-4](https://huggingface.co/CompVis/stable-diffusion-v1-4) or a path to a local directory, *e.g.*
"./stable-diffusion". To correctly retrieve which models and components should be loaded one has to provide a `model_index.json` file, *e.g.* [CompVis/stable-diffusion-v1-4/model_index.json](https://huggingface.co/CompVis/stable-diffusion-v1-4/blob/main/model_index.json), which defines all components that should be 
loaded into the pipelines. More specifically, for each model/component one needs to define the format `<name>: ["<library>", "<class name>"]`. `<name>` is the attribute name given to the loaded instance of `<class name>` which can be found in the library or pipeline folder called `"<library>"`.
- [`save_pretrained`](https://github.com/huggingface/diffusers/blob/5cbed8e0d157f65d3ddc2420dfd09f2df630e978/src/diffusers/pipeline_utils.py#L90) that accepts a local path, *e.g.* `./stable-diffusion` under which all models/components of the pipeline will be saved. For each component/model a folder is created inside the local path that is named after the given attribute name, *e.g.* `./stable_diffusion/unet`. 
In additon, a `model_index.json` file is created at the root of the local path, *e.g.* `./stable_diffusion/model_index.json` so that the complete pipeline can again be instantiated 
from the local path.
- [`to`](https://github.com/huggingface/diffusers/blob/5cbed8e0d157f65d3ddc2420dfd09f2df630e978/src/diffusers/pipeline_utils.py#L118) which accepts a `string` or `torch.device` to move all models that are of type `torch.nn.Module` to the passed device. The behavior is fully analogous to [PyTorch's `to` method](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.to).
- [`__call__`] method to use the pipeline in inference. The `__call__` defines infenence logic of the pipeline and should ideally encompass all parts from pre-processing to fowarding tensors to the different models and scheduler components as well as post-processing. The API of the `__call__` method can strongly vary from pipeline to pipeline. *E.g.* a text-to-image pipeline, such as [`StableDiffusionPipeline`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py) should accept among other things the text prompt to generate the image. A pure image generation pipeline, such as [DDPMPipeline](https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines/ddpm) on the other hand can be run without providing any inputs. To better understand what inputs can be adapted for 
each pipeline, one should look directly into the respective pipeline.

**Note**: All pipelines have PyTorch's autograd disabled by decorating the `__call__` method with a [`torch.no_grad`](https://pytorch.org/docs/stable/generated/torch.no_grad.html) decorator because pipelines should
not be used for training. If you want to store the gradients during the forward pass, we recommend writing your own pipeline, see also our [community-examples](https://github.com/huggingface/diffusers/tree/main/examples/commmunity)

## Contribution

As always, we are more than happy about any contribution
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
