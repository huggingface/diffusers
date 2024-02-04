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

## Hyperparameters

A short description of the consistency training-specific hyperparameters is as follows. The default hyperparameter values follow those in the improved consistency training column in Table 1 of [Improved Techniques for Training Consistency Models](https://arxiv.org/abs/2310.14189).

- Time Discretization
    - `sigma_min`/`sigma_max`: These define the lower and upper boundaries of the noise level $\sigma \in [\sigma_{min}, \sigma_{max}]$ By default, these are set to $\sigma_{min} = 0.002$ and $\sigma_{max} = 80.0$, following both the [original consistency models paper](https://arxiv.org/abs/2303.01469) and the [improved consistency training paper](https://arxiv.org/abs/2310.14189).
    - `rho`: in practice, the time interval $[\sigma_{min}, \sigma_{max}]$ is discretized into a sequence of noise levels $\sigma_{min} = \sigma_1 < \ldots < \sigma_{N} = \sigma_{max}$ following the Karras sigmas with parameter $\rho$:
    $$\sigma_i = (\sigma_{min}^{1 / \rho} - \frac{i + 1}{N - 1}(\sigma_{max}^{1 / \rho} - \sigma_{min}^{1 / \rho}))^\rho$$
    By default, $\rho = 7$, which is the value originally suggested in the [EDM paper](https://arxiv.org/abs/2206.00364) and used in the consistency model papers.
    - `discretization_s_0`/`discretization_s_1`: During training, we vary the number of discretization steps $N$ following a discretization curriculum $N(k)$ based on the current training step $k$ out of $K$ (`max_train_steps`) total:
    $$N(k) = \min{(s_02^{\lfloor k / K' \rfloor}, s_1)} + 1, K' = \lfloor\frac{K}{\log_{2}{\lfloor s_1 / s_0 \rfloor} + 1}\rfloor$$
    In this exponential curriculum, we start with $s_0 + 1$ discretization steps at the beginning of training, with the number of discretization steps $N$ doubling after a set number of training iterations until the maximum number of discretization steps $s_1 + 1$ is reached. By default, $s_0 = 10$ and $s_1 = 1280$, which are the values used in the [improved consistency training paper](https://arxiv.org/abs/2310.14189).
    - `constant_discretization_steps`: If set, disables the above discretization curriculum and uses a constant curriculum $N(k) = s_0 + 1$. This is useful for debugging.
- Input and Output Preconditioning
    - `input_precond_type`: this specifies how the $c_{in}(\sigma)$ input preconditioning parameter is calculated. By default, this is set to `'cm'`, which uses the input preconditioning from the original CM paper (which is also the original EDM input preconditioning) $c_{in}(\sigma) = 1 / \sqrt{\sigma^2 + \sigma_{data}^2}$. If `'none'` is specified, no input preconditioning will be used.
    - `noise_precond_type`: this specifies the function $c_{noise}(\sigma)$ which transforms discrete timesteps $\sigma_i$ for input into the consistency model U-Net. By default, this is set to `'cm'`, which uses the function $c_{noise}(\sigma) = 1000 \cdot \frac{1}{4}\log{(\sigma + 10^{-44})}$ from [the original consistency models repo](https://github.com/openai/consistency_models/blob/e32b69ee436d518377db86fb2127a3972d0d8716/cm/karras_diffusion.py#L346). The original EDM noise preconditioning function $c_{noise}(\sigma) = \frac{1}{4}\log{\sigma}$ can be used by setting this argument to `'edm'`. If `'none'` is specified, no noise preconditioning will be used.
- Noise Schedule
    - `p_mean`/`p_std`: the probability of sampling noise level $\sigma$ for training is distributed according to a lognormal distribution where $\log{\sigma} \sim \mathcal{N}(P_{mean}, P_{std}^2)$. Since we discretize the noise levels $\{\sigma_i\}$, we use a discretized version of the distribution where $i \sim p(i)$ and
    $$p(i) \propto \textrm{erf}{(\frac{\log{\sigma_{i + 1}} - P_{mean}}{\sqrt{2}P_{std}})} - \textrm{erf}{(\frac{\log{\sigma_{i}} - P_{mean}}{\sqrt{2}P_{std}})}$$
    By default, $P_{mean} = -1.1$ and $P_{std} = 2.0$, which are the default values used in the [improved consistency training paper](https://arxiv.org/abs/2310.14189).
- Loss
    - `huber_c`: this corresponds to the $c$ parameter in the Pseudo-Huber metric
    $$d(x, y) = \sqrt{\mid\mid x - y \mid\mid_2^2 + c^2} - c$$
    If not set, this will default to the heuristic value of $c = 0.00054\sqrt{d}$ where $d$ is dimensionality of the input image data suggested in the [improved consistency training paper](https://arxiv.org/abs/2310.14189).
- Exponential Moving Average (EMA)
    - `use_ema`: set this to use EMA for the student model (the model updated via gradient descent). Note that EMA is not used to update the teacher model (the model not updated via gradient descent with lower noise value); rather, the teacher parameters $\theta^-$ are set to the student parameters $\theta$ after each training step (equivalent to a EMA decay rate of 0).
    - `ema_min_decay`/`ema_max_decay`: specifies the minimum and maximum EMA decay. The [improved consistency training paper](https://arxiv.org/abs/2310.14189) uses a fixed EMA decay rate of `0.99993` for CIFAR10, which is achieved by the default setting of `ema_max_decay == 0.99993` and not setting `ema_min_decay` (when not set, `ema_min_decay` defaults to `ema_max_decay` so that the EMA decay is fixed throughout training).
