## Training an VQGAN VAE
VQVAEs were first introduced in [Neural Discrete Representation Learning](https://huggingface.co/papers/1711.00937) and was combined with a GAN in the paper [Taming Transformers for High-Resolution Image Synthesis](https://huggingface.co/papers/2012.09841). The basic idea of a VQVAE is it's a type of a variational auto encoder with tokens as the latent space similar to tokens for LLMs. This script was adapted from a [pr to huggingface's open-muse project](https://github.com/huggingface/open-muse/pull/52) with general code following [lucidrian's implementation of the vqgan training script](https://github.com/lucidrains/muse-maskgit-pytorch/blob/main/muse_maskgit_pytorch/trainers.py) but both of these implementation follow from the [taming transformer repo](https://github.com/CompVis/taming-transformers?tab=readme-ov-file).


Creating a training image set is [described in a different document](https://huggingface.co/docs/datasets/image_process#image-datasets).

### Installing the dependencies

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

### Training on CIFAR10

The command to train a VQGAN model on cifar10 dataset:

```bash
accelerate launch train_vqgan.py \
  --dataset_name=cifar10 \
  --image_column=img \
  --validation_images images/bird.jpg images/car.jpg images/dog.jpg images/frog.jpg \
  --resolution=128 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=8 \
  --report_to=wandb
```

An example training run is [here](https://wandb.ai/sayakpaul/vqgan-training/runs/0m5kzdfp) by @sayakpaul and a lower scale one [here](https://wandb.ai/dsbuddy27/vqgan-training/runs/eqd6xi4n?nw=nwuserisamu). The validation images can be obtained from [here](https://huggingface.co/datasets/diffusers/docs-images/tree/main/vqgan_validation_images).
The simplest way to improve the quality of a VQGAN model is to maximize the amount of information present in the bottleneck. The easiest way to do this is increasing the image resolution. However, other ways include, but not limited to, lowering compression by downsampling fewer times or increasing the vocabulary size which at most can be around 16384. How to do this is shown below.

# Modifying the architecture

To modify the architecture of the vqgan model you can save the config taken from [here](https://huggingface.co/kandinsky-community/kandinsky-2-2-decoder/blob/main/movq/config.json) and then provide that to the script with the option --model_config_name_or_path. This config is below
```
{
  "_class_name": "VQModel",
  "_diffusers_version": "0.17.0.dev0",
  "act_fn": "silu",
  "block_out_channels": [
    128,
    256,
    256,
    512
  ],
  "down_block_types": [
    "DownEncoderBlock2D",
    "DownEncoderBlock2D",
    "DownEncoderBlock2D",
    "AttnDownEncoderBlock2D"
  ],
  "in_channels": 3,
  "latent_channels": 4,
  "layers_per_block": 2,
  "norm_num_groups": 32,
  "norm_type": "spatial",
  "num_vq_embeddings": 16384,
  "out_channels": 3,
  "sample_size": 32,
  "scaling_factor": 0.18215,
  "up_block_types": [
    "AttnUpDecoderBlock2D",
    "UpDecoderBlock2D",
    "UpDecoderBlock2D",
    "UpDecoderBlock2D"
  ],
  "vq_embed_dim": 4
}
```
To lower the amount of layers in a VQGan, you can remove layers by modifying the block_out_channels, down_block_types, and up_block_types like below
```
{
  "_class_name": "VQModel",
  "_diffusers_version": "0.17.0.dev0",
  "act_fn": "silu",
  "block_out_channels": [
    128,
    256,
    256,
  ],
  "down_block_types": [
    "DownEncoderBlock2D",
    "DownEncoderBlock2D",
    "DownEncoderBlock2D",
  ],
  "in_channels": 3,
  "latent_channels": 4,
  "layers_per_block": 2,
  "norm_num_groups": 32,
  "norm_type": "spatial",
  "num_vq_embeddings": 16384,
  "out_channels": 3,
  "sample_size": 32,
  "scaling_factor": 0.18215,
  "up_block_types": [
    "UpDecoderBlock2D",
    "UpDecoderBlock2D",
    "UpDecoderBlock2D"
  ],
  "vq_embed_dim": 4
}
```
For increasing the size of the vocabularies you can increase num_vq_embeddings. However, [some research](https://magvit.cs.cmu.edu/v2/) shows that the representation of VQGANs start degrading after 2^14~16384 vq embeddings so it's not recommended to go past that.

## Extra training tips/ideas
During logging take care to make sure data_time is low. data_time is the amount spent loading the data and where the GPU is not active. So essentially, it's the time wasted. The easiest way to lower data time is to increase the --dataloader_num_workers to a higher number like 4. Due to a bug in Pytorch, this only works on linux based systems. For more details check [here](https://github.com/huggingface/diffusers/issues/7646)
Secondly, training should seem to be done when both the discriminator and the generator loss converges.
Thirdly, another low hanging fruit is just using ema using the --use_ema parameter. This tends to make the output images smoother. This has a con where you have to lower your batch size by 1 but it may be worth it.
Another more experimental low hanging fruit is changing from the vgg19 to different models for the lpips loss using the --timm_model_backend. If you do this, I recommend also changing the timm_model_layers parameter to the layer in your model which you think is best for representation. However, be careful with the feature map norms since this can easily overdominate the loss.