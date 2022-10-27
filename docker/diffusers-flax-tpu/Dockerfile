FROM ubuntu:20.04
LABEL maintainer="Hugging Face"
LABEL repository="diffusers"

RUN apt update && \
    apt install -y bash \
                   build-essential \
                   git \
                   git-lfs \
                   curl \
                   ca-certificates \
                   python3.8 \
                   python3-pip \
                   python3.8-venv && \
    rm -rf /var/lib/apt/lists

# TPU-specific steps
RUN curl -L https://storage.googleapis.com/cloud-tpu-tpuvm-artifacts/libtpu/1.4.0/libtpu.so -o /lib/libtpu.so
ENV TPU_NAME=local
ENV TF_CPP_MIN_LOG_LEVEL=0

# make sure to use venv
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# pre-install the heavy dependencies (these can later be overridden by the deps from setup.py)
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir \
        "jax[tpu]>=0.2.16,!=0.3.2,<=0.3.6" \
        -f https://storage.googleapis.com/jax-releases/libtpu_releases.html && \
    python3 -m pip install --no-cache-dir \
        "flax>=0.4.1" \
        "jaxlib>=0.1.65,<=0.3.6" && \
    python3 -m pip install --no-cache-dir \
        accelerate \
        datasets \
        hf-doc-builder \
        huggingface-hub \
        modelcards \
        numpy \
        scipy \
        tensorboard \
        transformers

CMD ["/bin/bash"]