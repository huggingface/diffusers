# Schedulers

- Schedulers are the algorithms to use diffusion models in inference as well as for training. They include the noise schedules and define algorithm-specific diffusion steps.
- Schedulers can be used interchangeable between diffusion models in inference to find the preferred trade-off between speed and generation quality.
- Schedulers are available in numpy, but can easily be transformed into PyTorch.

## API

- Schedulers should provide one or more `def step(...)` functions that should be called iteratively to unroll the diffusion loop during 
the forward pass.
- Schedulers should be framework specific.

## Examples

- The DDPM scheduler was proposed in [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) and can be found in [scheduling_ddpm.py](https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_ddpm.py). An example of how to use this scheduler can be found in [pipeline_ddpm.py](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/pipeline_ddpm.py).
- The DDIM scheduler was proposed in [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) and can be found in [scheduling_ddim.py](https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_ddim.py). An example of how to use this scheduler can be found in [pipeline_ddim.py](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/pipeline_ddim.py).
- The PNDM scheduler was proposed in [Pseudo Numerical Methods for Diffusion Models on Manifolds](https://arxiv.org/abs/2202.09778) and can be found in [scheduling_pndm.py](https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_pndm.py). An example of how to use this scheduler can be found in [pipeline_pndm.py](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/pipeline_pndm.py).
