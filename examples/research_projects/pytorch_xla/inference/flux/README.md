# Generating images using Flux and PyTorch/XLA

The `flux_inference` script shows how to do image generation using Flux on TPU devices using PyTorch/XLA. It uses the pallas kernel for flash attention for faster generation.

It has been tested on [Trillium](https://cloud.google.com/blog/products/compute/introducing-trillium-6th-gen-tpus) TPU versions. No other TPU types have been tested.

## Create TPU

To create a TPU on Google Cloud, follow [this guide](https://cloud.google.com/tpu/docs/v6e)

## Setup TPU environment

SSH into the VM and install Pytorch, Pytorch/XLA

```bash
pip install torch~=2.5.0 torch_xla[tpu]~=2.5.0 -f https://storage.googleapis.com/libtpu-releases/index.html -f https://storage.googleapis.com/libtpu-wheels/index.html
pip install torch_xla[pallas] -f https://storage.googleapis.com/jax-releases/jax_nightly_releases.html -f https://storage.googleapis.com/jax-releases/jaxlib_nightly_releases.html
```

Verify that PyTorch and PyTorch/XLA were installed correctly:

```bash
python3 -c "import torch; import torch_xla;"
```

Install dependencies

```bash
pip install transformers accelerate sentencepiece structlog
pushd ../../..
pip install .
popd
```

## Run the inference job

### Authenticate

Run the following command to authenticate your token in order to download Flux weights.

```bash
huggingface-cli login
```

Then run:

```bash
python flux_inference.py
```

The script loads the text encoders onto the CPU and the Flux transformer and VAE models onto the TPU. The first time the script runs, the compilation time is longer, while the cache stores the compiled programs. On subsequent runs, compilation is much faster and the subsequent passes being the fastest. 

On a Trillium v6e-4, you should expect ~9 sec / 4 images or 2.25 sec / image (as devices run generation in parallel):

```bash
WARNING:root:libtpu.so and TPU device found. Setting PJRT_DEVICE=TPU.
Loading checkpoint shards: 100%|███████████████████████████████| 2/2 [00:00<00:00,  7.01it/s]
Loading pipeline components...:  40%|██████████▍               | 2/5 [00:00<00:00,  3.78it/s]You set `add_prefix_space`. The tokenizer needs to be converted from the slow tokenizers
Loading pipeline components...: 100%|██████████████████████████| 5/5 [00:00<00:00,  6.72it/s]
2025-01-10 00:51:25 [info     ] loading flux from black-forest-labs/FLUX.1-dev
2025-01-10 00:51:25 [info     ] loading flux from black-forest-labs/FLUX.1-dev
2025-01-10 00:51:26 [info     ] loading flux from black-forest-labs/FLUX.1-dev
2025-01-10 00:51:26 [info     ] loading flux from black-forest-labs/FLUX.1-dev
Loading pipeline components...: 100%|██████████████████████████| 3/3 [00:00<00:00,  4.29it/s]
Loading pipeline components...: 100%|██████████████████████████| 3/3 [00:00<00:00,  3.26it/s]
Loading pipeline components...: 100%|██████████████████████████| 3/3 [00:00<00:00,  3.27it/s]
Loading pipeline components...: 100%|██████████████████████████| 3/3 [00:00<00:00,  3.25it/s]
2025-01-10 00:51:34 [info     ] starting compilation run...   
2025-01-10 00:51:35 [info     ] starting compilation run...   
2025-01-10 00:51:37 [info     ] starting compilation run...   
2025-01-10 00:51:37 [info     ] starting compilation run...   
2025-01-10 00:52:52 [info     ] compilation took 78.5155531649998 sec.
2025-01-10 00:52:53 [info     ] starting inference run...     
2025-01-10 00:52:57 [info     ] compilation took 79.52986721400157 sec.
2025-01-10 00:52:57 [info     ] compilation took 81.91776501700042 sec.
2025-01-10 00:52:57 [info     ] compilation took 80.24951512600092 sec.
2025-01-10 00:52:57 [info     ] starting inference run...     
2025-01-10 00:52:57 [info     ] starting inference run...     
2025-01-10 00:52:58 [info     ] starting inference run...     
2025-01-10 00:53:22 [info     ] inference time: 25.112665320000815
2025-01-10 00:53:30 [info     ] inference time: 7.7019307739992655
2025-01-10 00:53:38 [info     ] inference time: 7.693858365000779
2025-01-10 00:53:46 [info     ] inference time: 7.690621814001133
2025-01-10 00:53:53 [info     ] inference time: 7.679490454000188
2025-01-10 00:54:01 [info     ] inference time: 7.68949568500102
2025-01-10 00:54:09 [info     ] inference time: 7.686633744000574
2025-01-10 00:54:16 [info     ] inference time: 7.696786873999372
2025-01-10 00:54:24 [info     ] inference time: 7.691988694999964
2025-01-10 00:54:32 [info     ] inference time: 7.700649563999832
2025-01-10 00:54:39 [info     ] inference time: 7.684993574001055
2025-01-10 00:54:47 [info     ] inference time: 7.68343457499941
2025-01-10 00:54:55 [info     ] inference time: 7.667921153999487
2025-01-10 00:55:02 [info     ] inference time: 7.683585194001353
2025-01-10 00:55:06 [info     ] avg. inference over 15 iterations took 8.61202360273334 sec.
2025-01-10 00:55:07 [info     ] avg. inference over 15 iterations took 8.952725123600006 sec.
2025-01-10 00:55:10 [info     ] inference time: 7.673799695001435
2025-01-10 00:55:10 [info     ] avg. inference over 15 iterations took 8.849190365400379 sec.
2025-01-10 00:55:10 [info     ] saved metric information as /tmp/metrics_report.txt
2025-01-10 00:55:12 [info     ] avg. inference over 15 iterations took 8.940161458400205 sec.
```