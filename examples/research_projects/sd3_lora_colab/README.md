# Running Stable Diffusion 3 DreamBooth LoRA training under 16GB

This is an **EDUCATIONAL** project that provides utilities for DreamBooth LoRA training for [Stable Diffusion 3 (SD3)](ttps://huggingface.co/papers/2403.03206) under 16GB GPU VRAM. This means you can successfully try out this project using a [free-tier Colab Notebook](https://colab.research.google.com/github/huggingface/diffusers/blob/main/examples/research_projects/sd3_lora_colab/sd3_dreambooth_lora_16gb.ipynb) instance. 🤗

> [!NOTE]
> SD3 is gated, so you need to make sure you agree to [share your contact info](https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers) to access the model before using it with Diffusers. Once you have access, you need to log in so your system knows you’re authorized. Use the command below to log in:

```bash
huggingface-cli login
```

This will also allow us to push the trained model parameters to the Hugging Face Hub platform.

For setup, inference code, and details on how to run the code, please follow the Colab Notebook provided above. 

## How

We make use of several techniques to make this possible:

* Compute the embeddings from the instance prompt and serialize them for later reuse. This is implemented in the [`compute_embeddings.py`](./compute_embeddings.py) script. We use an 8bit (as introduced in [`LLM.int8()`](https://arxiv.org/abs/2208.07339)) T5 to reduce memory requirements to ~10.5GB. 
* In the `train_dreambooth_sd3_lora_miniature.py` script, we make use of:
  * 8bit Adam for optimization through the `bitsandbytes` library.
  * Gradient checkpointing and gradient accumulation.
  * FP16 precision.
  * Flash attention through `F.scaled_dot_product_attention()`. 

Computing the text embeddings is arguably the most memory-intensive part in the pipeline as SD3 employs three text encoders. If we run them in FP32, it will take about 20GB of VRAM. With FP16, we are down to 12GB. 


## Gotchas

This project is educational. It exists to showcase the possibility of fine-tuning a big diffusion system on consumer GPUs. But additional components might have to be added to obtain state-of-the-art performance. Below are some commonly known gotchas that users should be aware of:

* Training of text encoders is purposefully disabled. 
* Techniques such as prior-preservation is unsupported. 
* Custom instance captions for instance images are unsupported, but this should be relatively easy to integrate.

Hopefully, this project gives you a template to extend it further to suit your needs.
