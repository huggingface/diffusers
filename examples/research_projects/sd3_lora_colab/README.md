# Running Stable Diffusion 3 DreamBooth LoRA training under 16GB

This is **EDUCATIONAL** project that provides utilities to conduct DreamBooth LoRA training for [Stable Diffusion 3 (SD3)](ttps://huggingface.co/papers/2403.03206) under 16GB GPU VRAM. This means you can successfully try out this project using a free-tier Colab Notebook instance. Here is one for you to quickly get started (TODO: provide Colab link) 🤗

> [!NOTE]
> As SD3 is gated, before using it with diffusers you first need to go to the [Stable Diffusion 3 Medium Hugging Face page](https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers), fill in the form and accept the gate. Once you are in, you need to log in so that your system knows you’ve accepted the gate. Use the command below to log in:

```bash
huggingface-cli login
```

This will also allow us to push the trained model parameters to the Hugging Face Hub platform.

For setup, inference code, and details on how to run the code, please follow the Colab Notebook provided above. 

## How

We make use of several techniques to make this possible:

* Compute the embeddings from the instance prompt and serialize them for later reuse. This is implemented in the [`compute_embeddings.py`](./compute_embeddings.py) script. We use an 8bit T5 to keep memory requirements manageable. More details have been provided below.
* In the `train_dreambooth_sd3_lora_miniature.py` script, we make use of:
  * 8bit Adam for optimization through the `bitsandbytes` library.
  * Gradient checkpointing and gradient accumulation.
  * FP16 precision.
  * Flash attention through `F.scaled_dot_product_attention()`. 

Computing the text embeddings is arguably the most memory-intensive part in the pipeline as SD3 employs three text encoders. If we run them in FP32, it will take about 20GB of VRAM. With FP16, we are down to 12GB. 

For this project, we leverage 8Bit T5 (8bit as introduced in [`LLM.int8()`](https://arxiv.org/abs/2208.07339)) that reduces the memory requirements further to ~10.5GB.

## Gotchas

This project is educational. It exists to showcase the possibility of fine-tuning a big diffusion system on consumer GPUs. But additional components might have to be added to obtain state-of-the-art performance. Below are are some commonly known gotchas that the users should be aware of:

* Training of text encoders is purposefully disabled. 
* Techniques such as prior-preservation is unsupported. 
* Custom instance captions for instance images are unsupported. But this should be relatively easy to integrate.

Hopefully, this project gives you a template to extend it further to suit your needs.