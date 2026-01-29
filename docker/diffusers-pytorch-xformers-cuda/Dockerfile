FROM nvidia/cuda:12.9.1-runtime-ubuntu24.04
LABEL maintainer="Hugging Face"
LABEL repository="diffusers"

ARG PYTHON_VERSION=3.10
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get -y update \
    && apt-get install -y software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update

RUN apt install -y bash \
    build-essential \
    git \
    git-lfs \
    curl \
    ca-certificates \
    libglib2.0-0 \
    libsndfile1-dev \
    libgl1 \
    python3 \
    python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"
ENV VIRTUAL_ENV="/opt/venv"
ENV UV_PYTHON_INSTALL_DIR=/opt/uv/python
RUN uv venv --python ${PYTHON_VERSION} --seed ${VIRTUAL_ENV}
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# pre-install the heavy dependencies (these can later be overridden by the deps from setup.py)
# Install torch, torchvision, and torchaudio together to ensure compatibility
RUN uv pip install --no-cache-dir \
    torch \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cu129

# Install compatible versions of numba/llvmlite for Python 3.10+
RUN uv pip install --no-cache-dir \
    "llvmlite>=0.40.0" \
    "numba>=0.57.0"

RUN uv pip install --no-cache-dir "git+https://github.com/huggingface/diffusers.git@main#egg=diffusers[test]"

# Extra dependencies
RUN uv pip install --no-cache-dir \
    accelerate \
    numpy==1.26.4 \
    pytorch-lightning \
    hf_xet \
    xformers

CMD ["/bin/bash"]
