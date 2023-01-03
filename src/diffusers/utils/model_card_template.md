---
{{ card_data }}
---

<!-- This model card has been generated automatically according to the information the training script had access to. You
should probably proofread and complete it, then remove this comment. -->

# {{ model_name | default("Diffusion Model") }}

## Model description

This diffusion model is trained with the [🤗 Diffusers](https://github.com/huggingface/diffusers) library 
on the `{{ dataset_name }}` dataset.

## Intended uses & limitations

#### How to use

```python
# TODO: add an example code snippet for running this diffusion pipeline
```

#### Limitations and bias

[TODO: provide examples of latent issues and potential remediations]

## Training data

[TODO: describe the data used to train the model]

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: {{ learning_rate }}
- train_batch_size: {{ train_batch_size }}
- eval_batch_size: {{ eval_batch_size }}
- gradient_accumulation_steps: {{ gradient_accumulation_steps }}
- optimizer: AdamW with betas=({{ adam_beta1 }}, {{ adam_beta2 }}), weight_decay={{ adam_weight_decay }} and epsilon={{ adam_epsilon }}
- lr_scheduler: {{ lr_scheduler }}
- lr_warmup_steps: {{ lr_warmup_steps }}
- ema_inv_gamma: {{ ema_inv_gamma }}
- ema_inv_gamma: {{ ema_power }}
- ema_inv_gamma: {{ ema_max_decay }}
- mixed_precision: {{ mixed_precision }}

### Training results

📈 [TensorBoard logs](https://huggingface.co/{{ repo_name }}/tensorboard?#scalars)


