# Stable Diffusion text-to-image fine-tuning

[DreamBooth](https://arxiv.org/abs/2208.12242) is a method to personalize text2image models like stable diffusion given just a few(3~5) images of a subject.
The `train_dreambooth.py` script shows how to implement the training procedure and adapt it for stable diffusion.


## Running locally 
### Installing the dependencies

Before running the scripts, make sure to install the library's training dependencies:

```bash
pip install git+https://github.com/huggingface/diffusers.git
pip install -U -r requirements.txt
```

And initialize an [🤗Accelerate](https://github.com/huggingface/accelerate/) environment with:

```bash
accelerate config
```