# Control-LoRA inference example

Control-LoRA is introduced by Stability AI in [stabilityai/control-lora](https://huggingface.co/stabilityai/control-lora) by adding low-rank parameter efficient fine tuning to ControlNet. This approach offers a more efficient and compact method to bring model control to a wider variety of consumer GPUs.

## Installing the dependencies

Before running the scripts, make sure to install the library's training dependencies:

**Important**

To make sure you can successfully run the latest versions of the example scripts, we highly recommend **installing from source** and keeping the install up to date as we update the example scripts frequently and install some example-specific requirements. To do this, execute the following steps in a new virtual environment:
```bash
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install .
```

Then cd in the example folder  and run
```bash
pip install -r requirements.txt
```

And initialize an [🤗Accelerate](https://github.com/huggingface/accelerate/) environment with:

```bash
accelerate config
```

## Inference on SDXL

[stabilityai/control-lora](https://huggingface.co/stabilityai/control-lora) provides a set of Control-LoRA weights for SDXL. Here we use the `canny` condition to generate an image from a text prompt and a reference image.

```bash
python control_lora.py
```

## Acknowledgements

- [stabilityai/control-lora](https://huggingface.co/stabilityai/control-lora)
- [comfyanonymous/ControlNet-v1-1_fp16_safetensors](https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors)
- [HighCWu/control-lora-v2](https://github.com/HighCWu/control-lora-v2)