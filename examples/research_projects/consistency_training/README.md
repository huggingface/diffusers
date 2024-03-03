# Consistency Training

`train_cm_ct_unconditional.py` trains a consistency model (CM) from scratch following the consistency training (CT) algorithm introduced in [Consistency Models](https://arxiv.org/abs/2303.01469) and refined in [Improved Techniques for Training Consistency Models](https://arxiv.org/abs/2310.14189). Both unconditional and class-conditional training are supported.

A usage example is as follows:

```bash
accelerate launch examples/research_projects/consistency_training/train_cm_ct_unconditional.py \
    --dataset_name="cifar10" \
    --dataset_image_column_name="img" \
    --output_dir="/path/to/output/dir" \
    --mixed_precision=fp16 \
    --resolution=32 \
    --max_train_steps=1000 --max_train_samples=10000 \
    --dataloader_num_workers=8 \
    --noise_precond_type="cm" --input_precond_type="cm" \
    --train_batch_size=4 \
    --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
    --use_8bit_adam \
    --use_ema \
    --validation_steps=100 --eval_batch_size=4 \
    --checkpointing_steps=100 --checkpoints_total_limit=10 \
    --class_conditional --num_classes=10 \
```