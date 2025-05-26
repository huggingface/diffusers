## Amused training

Amused can be finetuned on simple datasets relatively cheaply and quickly. Using 8bit optimizers, lora, and gradient accumulation, amused can be finetuned with as little as 5.5 GB. Here are a set of examples for finetuning amused on some relatively simple datasets. These training recipes are aggressively oriented towards minimal resources and fast verification -- i.e. the batch sizes are quite low and the learning rates are quite high. For optimal quality, you will probably want to increase the batch sizes and decrease learning rates.

All training examples use fp16 mixed precision and gradient checkpointing. We don't show 8 bit adam + lora as its about the same memory use as just using lora (bitsandbytes uses full precision optimizer states for weights below a minimum size).

### Finetuning the 256 checkpoint

These examples finetune on this [nouns](https://huggingface.co/datasets/m1guelpf/nouns) dataset.

Example results:

![noun1](https://huggingface.co/datasets/diffusers/docs-images/resolve/main/amused/noun1.png) ![noun2](https://huggingface.co/datasets/diffusers/docs-images/resolve/main/amused/noun2.png) ![noun3](https://huggingface.co/datasets/diffusers/docs-images/resolve/main/amused/noun3.png)


#### Full finetuning

Batch size: 8, Learning rate: 1e-4, Gives decent results in 750-1000 steps

| Batch Size | Gradient Accumulation Steps | Effective Total Batch Size | Memory Used |
|------------|-----------------------------|------------------|-------------|
|    8        |          1                   |     8             |      19.7 GB       |
|    4        |          2                   |     8             |      18.3 GB       |
|    1        |          8                   |     8             |      17.9 GB       |

```sh
accelerate launch train_amused.py \
    --output_dir <output path> \
    --train_batch_size <batch size> \
    --gradient_accumulation_steps <gradient accumulation steps> \
    --learning_rate 1e-4 \
    --pretrained_model_name_or_path amused/amused-256 \
    --instance_data_dataset  'm1guelpf/nouns' \
    --image_key image \
    --prompt_key text \
    --resolution 256 \
    --mixed_precision fp16 \
    --lr_scheduler constant \
    --validation_prompts \
        'a pixel art character with square red glasses, a baseball-shaped head and a orange-colored body on a dark background' \
        'a pixel art character with square orange glasses, a lips-shaped head and a red-colored body on a light background' \
        'a pixel art character with square blue glasses, a microwave-shaped head and a purple-colored body on a sunny background' \
        'a pixel art character with square red glasses, a baseball-shaped head and a blue-colored body on an orange background' \
        'a pixel art character with square red glasses' \
        'a pixel art character' \
        'square red glasses on a pixel art character' \
        'square red glasses on a pixel art character with a baseball-shaped head' \
    --max_train_steps 10000 \
    --checkpointing_steps 500 \
    --validation_steps 250 \
    --gradient_checkpointing
```

#### Full finetuning + 8 bit adam

Note that this training config keeps the batch size low and the learning rate high to get results fast with low resources. However, due to 8 bit adam, it will diverge eventually. If you want to train for longer, you will have to up the batch size and lower the learning rate.

Batch size: 16, Learning rate: 2e-5, Gives decent results in ~750 steps

| Batch Size | Gradient Accumulation Steps | Effective Total Batch Size | Memory Used |
|------------|-----------------------------|------------------|-------------|
|    16        |          1                   |     16             |      20.1 GB       |
|    8        |          2                   |      16           |      15.6 GB       |
|    1        |          16                   |     16            |      10.7 GB       |

```sh
accelerate launch train_amused.py \
    --output_dir <output path> \
    --train_batch_size <batch size> \
    --gradient_accumulation_steps <gradient accumulation steps> \
    --learning_rate 2e-5 \
    --use_8bit_adam \
    --pretrained_model_name_or_path amused/amused-256 \
    --instance_data_dataset  'm1guelpf/nouns' \
    --image_key image \
    --prompt_key text \
    --resolution 256 \
    --mixed_precision fp16 \
    --lr_scheduler constant \
    --validation_prompts \
        'a pixel art character with square red glasses, a baseball-shaped head and a orange-colored body on a dark background' \
        'a pixel art character with square orange glasses, a lips-shaped head and a red-colored body on a light background' \
        'a pixel art character with square blue glasses, a microwave-shaped head and a purple-colored body on a sunny background' \
        'a pixel art character with square red glasses, a baseball-shaped head and a blue-colored body on an orange background' \
        'a pixel art character with square red glasses' \
        'a pixel art character' \
        'square red glasses on a pixel art character' \
        'square red glasses on a pixel art character with a baseball-shaped head' \
    --max_train_steps 10000 \
    --checkpointing_steps 500 \
    --validation_steps 250 \
    --gradient_checkpointing
```

#### Full finetuning + lora

Batch size: 16, Learning rate: 8e-4, Gives decent results in 1000-1250 steps

| Batch Size | Gradient Accumulation Steps | Effective Total Batch Size | Memory Used |
|------------|-----------------------------|------------------|-------------|
|    16        |          1                   |     16             |      14.1 GB       |
|    8        |          2                   |      16           |      10.1 GB       |
|    1        |          16                   |     16            |      6.5 GB       |

```sh
accelerate launch train_amused.py \
    --output_dir <output path> \
    --train_batch_size <batch size> \
    --gradient_accumulation_steps <gradient accumulation steps> \
    --learning_rate 8e-4 \
    --use_lora \
    --pretrained_model_name_or_path amused/amused-256 \
    --instance_data_dataset  'm1guelpf/nouns' \
    --image_key image \
    --prompt_key text \
    --resolution 256 \
    --mixed_precision fp16 \
    --lr_scheduler constant \
    --validation_prompts \
        'a pixel art character with square red glasses, a baseball-shaped head and a orange-colored body on a dark background' \
        'a pixel art character with square orange glasses, a lips-shaped head and a red-colored body on a light background' \
        'a pixel art character with square blue glasses, a microwave-shaped head and a purple-colored body on a sunny background' \
        'a pixel art character with square red glasses, a baseball-shaped head and a blue-colored body on an orange background' \
        'a pixel art character with square red glasses' \
        'a pixel art character' \
        'square red glasses on a pixel art character' \
        'square red glasses on a pixel art character with a baseball-shaped head' \
    --max_train_steps 10000 \
    --checkpointing_steps 500 \
    --validation_steps 250 \
    --gradient_checkpointing
```

### Finetuning the 512 checkpoint

These examples finetune on this [minecraft](https://huggingface.co/monadical-labs/minecraft-preview) dataset.

Example results:

![minecraft1](https://huggingface.co/datasets/diffusers/docs-images/resolve/main/amused/minecraft1.png) ![minecraft2](https://huggingface.co/datasets/diffusers/docs-images/resolve/main/amused/minecraft2.png) ![minecraft3](https://huggingface.co/datasets/diffusers/docs-images/resolve/main/amused/minecraft3.png)

#### Full finetuning

Batch size: 8, Learning rate: 8e-5, Gives decent results in 500-1000 steps

| Batch Size | Gradient Accumulation Steps | Effective Total Batch Size | Memory Used |
|------------|-----------------------------|------------------|-------------|
|    8        |          1                   |     8             |      24.2 GB       |
|    4        |          2                   |     8             |      19.7 GB       |
|    1        |          8                   |     8             |      16.99 GB       |

```sh
accelerate launch train_amused.py \
    --output_dir <output path> \
    --train_batch_size <batch size> \
    --gradient_accumulation_steps <gradient accumulation steps> \
    --learning_rate 8e-5 \
    --pretrained_model_name_or_path amused/amused-512 \
    --instance_data_dataset  'monadical-labs/minecraft-preview' \
    --prompt_prefix 'minecraft ' \
    --image_key image \
    --prompt_key text \
    --resolution 512 \
    --mixed_precision fp16 \
    --lr_scheduler constant \
    --validation_prompts \
        'minecraft Avatar' \
        'minecraft character' \
        'minecraft' \
        'minecraft president' \
        'minecraft pig' \
    --max_train_steps 10000 \
    --checkpointing_steps 500 \
    --validation_steps 250 \
    --gradient_checkpointing
```

#### Full finetuning + 8 bit adam

Batch size: 8, Learning rate: 5e-6, Gives decent results in 500-1000 steps

| Batch Size | Gradient Accumulation Steps | Effective Total Batch Size | Memory Used |
|------------|-----------------------------|------------------|-------------|
|    8        |          1                   |     8             |      21.2 GB       |
|    4        |          2                   |     8             |      13.3 GB       |
|    1        |          8                   |     8             |      9.9 GB       |

```sh
accelerate launch train_amused.py \
    --output_dir <output path> \
    --train_batch_size <batch size> \
    --gradient_accumulation_steps <gradient accumulation steps> \
    --learning_rate 5e-6 \
    --pretrained_model_name_or_path amused/amused-512 \
    --instance_data_dataset  'monadical-labs/minecraft-preview' \
    --prompt_prefix 'minecraft ' \
    --image_key image \
    --prompt_key text \
    --resolution 512 \
    --mixed_precision fp16 \
    --lr_scheduler constant \
    --validation_prompts \
        'minecraft Avatar' \
        'minecraft character' \
        'minecraft' \
        'minecraft president' \
        'minecraft pig' \
    --max_train_steps 10000 \
    --checkpointing_steps 500 \
    --validation_steps 250 \
    --gradient_checkpointing
```

#### Full finetuning + lora

Batch size: 8, Learning rate: 1e-4, Gives decent results in 500-1000 steps

| Batch Size | Gradient Accumulation Steps | Effective Total Batch Size | Memory Used |
|------------|-----------------------------|------------------|-------------|
|    8        |          1                   |     8             |      12.7 GB       |
|    4        |          2                   |     8             |      9.0 GB       |
|    1        |          8                   |     8             |      5.6 GB       |

```sh
accelerate launch train_amused.py \
    --output_dir <output path> \
    --train_batch_size <batch size> \
    --gradient_accumulation_steps <gradient accumulation steps> \
    --learning_rate 1e-4 \
    --use_lora \
    --pretrained_model_name_or_path amused/amused-512 \
    --instance_data_dataset  'monadical-labs/minecraft-preview' \
    --prompt_prefix 'minecraft ' \
    --image_key image \
    --prompt_key text \
    --resolution 512 \
    --mixed_precision fp16 \
    --lr_scheduler constant \
    --validation_prompts \
        'minecraft Avatar' \
        'minecraft character' \
        'minecraft' \
        'minecraft president' \
        'minecraft pig' \
    --max_train_steps 10000 \
    --checkpointing_steps 500 \
    --validation_steps 250 \
    --gradient_checkpointing
```

### Styledrop

[Styledrop](https://huggingface.co/papers/2306.00983) is an efficient finetuning method for learning a new style from just one or very few images. It has an optional first stage to generate human picked additional training samples. The additional training samples can be used to augment the initial images. Our examples exclude the optional additional image selection stage and instead we just finetune on a single image.

This is our example style image:
![example](https://huggingface.co/datasets/diffusers/docs-images/resolve/main/amused/A%20mushroom%20in%20%5BV%5D%20style.png)

Download it to your local directory with
```sh
wget https://huggingface.co/datasets/diffusers/docs-images/resolve/main/amused/A%20mushroom%20in%20%5BV%5D%20style.png
```

#### 256

Example results:

![glowing_256_1](https://huggingface.co/datasets/diffusers/docs-images/resolve/main/amused/glowing_256_1.png) ![glowing_256_2](https://huggingface.co/datasets/diffusers/docs-images/resolve/main/amused/glowing_256_2.png) ![glowing_256_3](https://huggingface.co/datasets/diffusers/docs-images/resolve/main/amused/glowing_256_3.png)

Learning rate: 4e-4, Gives decent results in 1500-2000 steps

Memory used: 6.5 GB

```sh
accelerate launch train_amused.py \
    --output_dir <output path> \
    --mixed_precision fp16 \
    --report_to wandb \
    --use_lora \
    --pretrained_model_name_or_path amused/amused-256 \
    --train_batch_size 1 \
    --lr_scheduler constant \
    --learning_rate 4e-4 \
    --validation_prompts \
        'A chihuahua walking on the street in [V] style' \
        'A banana on the table in [V] style' \
        'A church on the street in [V] style' \
        'A tabby cat walking in the forest in [V] style' \
    --instance_data_image 'A mushroom in [V] style.png' \
    --max_train_steps 10000 \
    --checkpointing_steps 500 \
    --validation_steps 100 \
    --resolution 256
```

#### 512

Example results:

![glowing_512_1](https://huggingface.co/datasets/diffusers/docs-images/resolve/main/amused/glowing_512_1.png) ![glowing_512_2](https://huggingface.co/datasets/diffusers/docs-images/resolve/main/amused/glowing_512_2.png) ![glowing_512_3](https://huggingface.co/datasets/diffusers/docs-images/resolve/main/amused/glowing_512_3.png)

Learning rate: 1e-3, Lora alpha 1, Gives decent results in 1500-2000 steps

Memory used: 5.6 GB

```
accelerate launch train_amused.py \
    --output_dir <output path> \
    --mixed_precision fp16 \
    --report_to wandb \
    --use_lora \
    --pretrained_model_name_or_path amused/amused-512 \
    --train_batch_size 1 \
    --lr_scheduler constant \
    --learning_rate 1e-3 \
    --validation_prompts \
        'A chihuahua walking on the street in [V] style' \
        'A banana on the table in [V] style' \
        'A church on the street in [V] style' \
        'A tabby cat walking in the forest in [V] style' \
    --instance_data_image 'A mushroom in [V] style.png' \
    --max_train_steps 100000 \
    --checkpointing_steps 500 \
    --validation_steps 100 \
    --resolution 512 \
    --lora_alpha 1
```