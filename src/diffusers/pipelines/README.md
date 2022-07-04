# Pipelines

- Pipelines are a collection of end-to-end diffusion systems that can be used out-of-the-box
- Pipelines should stay as close as possible to their original implementation 
- Pipelines can include components of other library, such as text-encoders. 

## API

TODO(Patrick, Anton, Suraj)

## Examples

- DDPM for unconditional image generation in [pipeline_ddpm](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/ddpm/pipeline_ddpm.py).
- DDIM for unconditional image generation in [pipeline_ddim](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/ddim/pipeline_ddim.py).
- PNDM for unconditional image generation in [pipeline_pndm](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/pndm/pipeline_pndm.py).
- Latent diffusion for text to image generation / conditional image generation in [pipeline_latent_diffusion](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/latent_diffusion/pipeline_latent_diffusion.py).
- Glide for text to image generation / conditional image generation in [pipeline_glide](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/glide/pipeline_glide.py).
- BDDMPipeline for spectrogram-to-sound vocoding in [pipeline_bddm](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/bddm/pipeline_bddm.py).
- Grad-TTS for text to audio generation / conditional audio generation in [pipeline_grad_tts](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/grad_tts/pipeline_grad_tts.py).
