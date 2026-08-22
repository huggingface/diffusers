# Prebuilt image for `diffusers-cli run --remote`, so a sandbox starts straight into the run
# instead of installing the dependency set on every cold boot.
#
# The base is the same image the CLI used to boot into, so the torch + CUDA layer is unchanged:
# cuda12.8 is the highest cuda12.x tag below the HF Jobs host driver's CUDA 12.9 max.
#
# The Python deps below mirror `_DEFAULT_REMOTE_DEPS` in `src/diffusers/commands/run.py` —
# keep the two lists in sync.
FROM pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime
LABEL maintainer="Hugging Face"
LABEL repository="diffusers"

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
# Weight loading dominates what is left of the startup cost. Xet-accelerated downloads come from
# the `huggingface_hub[hf_xet]` install below (on by default once `hf_xet` is present); parallel
# shard loading is opt-in, and the sandbox env at creation time can still override it.
ENV HF_ENABLE_PARALLEL_LOADING=1

# libgl1/libglib2.0-0 back PIL + the cv2 video fallback; libsndfile1 backs torchaudio decoding.
RUN apt-get -y update && apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    curl \
    git \
    git-lfs \
    libgl1 \
    libglib2.0-0 \
    libsndfile1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# torch, torchvision, and torchaudio come from the base image as a matching set — never reinstall
# them here, an ABI mismatch breaks the torchvision/torchaudio C++ extensions. Fail the build
# instead if that assumption ever stops holding.
RUN python -c "import torch, torchvision, torchaudio"

# `uv pip install --system` targets the base image's Python — Ubuntu's system interpreter, which
# is where torch already lives and where the CLI installs `--dependencies` extras at runtime.
# --break-system-packages bypasses the PEP 668 marker that interpreter carries; harmless in a
# single-purpose image, and it matches what the CLI does inside the sandbox.
RUN uv pip install --system --break-system-packages --no-cache-dir \
    "huggingface_hub[hf_xet]>=1.23" \
    accelerate \
    transformers \
    safetensors \
    sentencepiece \
    ftfy \
    peft \
    imageio \
    imageio-ffmpeg

RUN uv pip install --system --break-system-packages --no-cache-dir "git+https://github.com/huggingface/diffusers.git@main#egg=diffusers"

# Smoke-test the console script the sandbox invokes, so a broken install fails the build.
RUN diffusers-cli env

CMD ["/bin/bash"]
