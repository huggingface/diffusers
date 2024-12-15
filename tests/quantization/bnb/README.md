The tests here are adapted from [`transformers` tests](https://github.com/huggingface/transformers/tree/409fcfdfccde77a14b7cc36972b774cabc371ae1/tests/quantization/bnb).

They were conducted on the `audace` machine, using a single RTX 4090. Below is `nvidia-smi`:

```bash
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.90.07              Driver Version: 550.90.07      CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 4090        Off |   00000000:01:00.0 Off |                  Off |
| 30%   55C    P0             61W /  450W |       1MiB /  24564MiB |      2%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   1  NVIDIA GeForce RTX 4090        Off |   00000000:13:00.0 Off |                  Off |
| 30%   51C    P0             60W /  450W |       1MiB /  24564MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
```

`diffusers-cli`:

```bash
- 🤗 Diffusers version: 0.31.0.dev0
- Platform: Linux-5.15.0-117-generic-x86_64-with-glibc2.35
- Running on Google Colab?: No
- Python version: 3.10.12
- PyTorch version (GPU?): 2.5.0.dev20240818+cu124 (True)
- Flax version (CPU?/GPU?/TPU?): not installed (NA)
- Jax version: not installed
- JaxLib version: not installed
- Huggingface_hub version: 0.24.5
- Transformers version: 4.44.2
- Accelerate version: 0.34.0.dev0
- PEFT version: 0.12.0
- Bitsandbytes version: 0.43.3
- Safetensors version: 0.4.4
- xFormers version: not installed
- Accelerator: NVIDIA GeForce RTX 4090, 24564 MiB
NVIDIA GeForce RTX 4090, 24564 MiB
- Using GPU in script?: Yes
```