FROM python:3.10-slim
ENV PYTHONDONTWRITEBYTECODE=1
LABEL maintainer="Hugging Face"
LABEL repository="diffusers"

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get -y update && apt-get install -y bash \
    build-essential \
    git \
    git-lfs \
    curl \
    ca-certificates \
    libsndfile1-dev \
    libgl1

ENV UV_PYTHON=/usr/local/bin/python

# pre-install the heavy dependencies (these can later be overridden by the deps from setup.py)
RUN pip install --no-cache-dir --upgrade uv \
    uv pip install --no-cache-dir \
        torch \
        torchvision \
        torchaudio \
        invisible_watermark \
        --extra-index-url https://download.pytorch.org/whl/cpu && \
    uv pip install --no-cache-dir \
        accelerate \
        datasets \
        hf-doc-builder \
        huggingface-hub \
        Jinja2 \
        librosa \
        numpy==1.26.4 \
        scipy \
        tensorboard \
        transformers matplotlib  \
        hf_transfer

CMD ["/bin/bash"]
