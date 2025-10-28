The tests here are adapted from [`transformers` tests](https://github.com/huggingface/transformers/blob/3a8eb74668e9c2cc563b2f5c62fac174797063e0/tests/quantization/torchao_integration/).

The benchmarks were run on a single H100. Below is `nvidia-smi`:

```bash
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.104.12             Driver Version: 535.104.12   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA H100 80GB HBM3          On  | 00000000:53:00.0 Off |                    0 |
| N/A   34C    P0              69W / 700W |      2MiB / 81559MiB |      0%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|  No running processes found                                                           |
+---------------------------------------------------------------------------------------+
```

The benchmark results for Flux and CogVideoX can be found in [this](https://github.com/huggingface/diffusers/pull/10009) PR.

The tests, and the expected slices, were obtained from the `aws-g6e-xlarge-plus` GPU test runners. To run the slow tests, use the following command or an equivalent:

```bash
HF_XET_HIGH_PERFORMANCE=1 RUN_SLOW=1 pytest -s tests/quantization/torchao/test_torchao.py::SlowTorchAoTests
```

`diffusers-cli`:

```bash
- 🤗 Diffusers version: 0.32.0.dev0
- Platform: Linux-5.15.0-1049-aws-x86_64-with-glibc2.31
- Running on Google Colab?: No
- Python version: 3.10.14
- PyTorch version (GPU?): 2.6.0.dev20241112+cu121 (False)
- Flax version (CPU?/GPU?/TPU?): not installed (NA)
- Jax version: not installed
- JaxLib version: not installed
- Huggingface_hub version: 0.26.2
- Transformers version: 4.46.3
- Accelerate version: 1.1.1
- PEFT version: not installed
- Bitsandbytes version: not installed
- Safetensors version: 0.4.5
- xFormers version: not installed
```
