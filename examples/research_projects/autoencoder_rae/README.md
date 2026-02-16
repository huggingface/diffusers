# Training AutoencoderRAE

This example trains the decoder of `AutoencoderRAE` (stage-1 style), while keeping the representation encoder frozen.

It follows the same high-level training recipe as the official RAE stage-1 setup:
- frozen encoder
- train decoder
- pixel reconstruction loss
- optional encoder feature consistency loss

## Quickstart

```bash
accelerate launch examples/research_projects/autoencoder_rae/train_autoencoder_rae.py \
  --train_data_dir /path/to/imagenet_like_folder \
  --output_dir /tmp/autoencoder-rae \
  --resolution 256 \
  --encoder_cls dinov2 \
  --encoder_input_size 224 \
  --patch_size 16 \
  --image_size 256 \
  --decoder_hidden_size 1152 \
  --decoder_num_hidden_layers 28 \
  --decoder_num_attention_heads 16 \
  --decoder_intermediate_size 4096 \
  --train_batch_size 8 \
  --learning_rate 1e-4 \
  --num_train_epochs 10 \
  --report_to wandb \
  --reconstruction_loss_type l1 \
  --use_encoder_loss \
  --encoder_loss_weight 0.1
```

Note: stage-1 reconstruction loss assumes matching target/output spatial size, so `--resolution` must equal `--image_size`.

Dataset format is expected to be `ImageFolder`-compatible:

```text
train_data_dir/
  class_a/
    img_0001.jpg
  class_b/
    img_0002.jpg
```
