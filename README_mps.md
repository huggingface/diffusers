## How to use Stable Diffusion in Apple Silicon (M1/M2)

🤗 Diffusers is compatible with Apple silicon for Stable Diffusion inference, using the PyTorch `mps` device. These are the steps you need to follow to use your M1 or M2 computer with Stable Diffusion.

### Requirements

- Mac computer with Apple silicon (M1/M2) hardware.
- macOS 12.3 or later.
- arm64 version of Python.
- PyTorch [Preview (Nightly)](https://pytorch.org/get-started/locally/), version `1.13.0.dev20220830` or later.

### Inference Pipeline

The snippet shown below demonstrates how to use the `mps` backend using the familiar `to()` interface to move the Stable Diffusion pipeline to your M1 or M2 device.

We recommend to "prime" the pipeline using an additional one-time pass through it. This is a temporary workaround for a weird issue we have detected: the first inference pass produces slightly different results than subsequent ones. You only need to do this pass once, and it's ok to use just one inference step and discard the result.

```python
# make sure you're logged in with `huggingface-cli login`
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=True)
pipe = pipe.to("mps")

prompt = "a photo of an astronaut riding a horse on mars"

# First-time "warmup" pass (see explanation above)
_ = pipe(prompt, num_inference_steps=1)

# Results match those from the CPU device after the warmup pass.
image = pipe(prompt).images[0]
```

### Performance

These are the results we got on a M1 Max MacBook Pro with 64 GB of RAM, running macOS Ventura Version 13.0 Beta (22A5331f). We performed Stable Diffusion text-to-image generation of the same prompt for 50 inference steps, using a guidance scale of 7.5.

| Device | Steps | Time    |
|--------|-------|---------|
| CPU    | 50    | 213.46s |
| MPS    | 50    | 30.81s  |
