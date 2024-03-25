## Training an VQGAN VAE

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

The simplest way to improve the quality of a VQGAN model is to maximize the amount of information present in the bottleneck. The easiest way to do this is increasing the image resolution. However, other ways include, but not limited to, lowering compression by downsampling fewer times or increasing the vocaburary size which at most can be around 16384. How to do this is shown below.

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
For increasing the size of the vocaburaries you can increase num_vq_embeddings. However, [some research](https://magvit.cs.cmu.edu/v2/) shows that the representation of VQGANs start degrading after 2^14~16384 vq embeddings so it's not recommended to go past that.