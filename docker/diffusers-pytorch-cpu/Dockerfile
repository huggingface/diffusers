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
    libglib2.0-0 \
    libsndfile1-dev \
    libgl1

ENV UV_PYTHON=/usr/local/bin/python

# pre-install the heavy dependencies (these can later be overridden by the deps from setup.py)
RUN pip install uv
RUN uv pip install --no-cache-dir \
    torch \
    torchvision \
    torchaudio \
    --extra-index-url https://download.pytorch.org/whl/cpu

RUN uv pip install --no-cache-dir "git+https://github.com/huggingface/diffusers.git@main#egg=diffusers[test]"

# Extra dependencies
RUN uv pip install --no-cache-dir \
    accelerate \
    numpy==1.26.4 \
    hf_xet

RUN apt-get clean && rm -rf /var/lib/apt/lists/* && apt-get autoremove && apt-get autoclean

CMD ["/bin/bash"]
