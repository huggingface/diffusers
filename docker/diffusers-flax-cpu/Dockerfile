FROM ubuntu:20.04
LABEL maintainer="Hugging Face"
LABEL repository="diffusers"

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get -y update \
    && apt-get install -y software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa

RUN apt install -y bash \
        build-essential \
        git \
        git-lfs \
        curl \
        ca-certificates \
        libsndfile1-dev \
        libgl1 \
        python3.10 \
        python3-pip \
        python3.10-venv && \
    rm -rf /var/lib/apt/lists

# make sure to use venv
RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# pre-install the heavy dependencies (these can later be overridden by the deps from setup.py)
# follow the instructions here: https://cloud.google.com/tpu/docs/run-in-container#train_a_jax_model_in_a_docker_container
RUN python3 -m pip install --no-cache-dir --upgrade pip uv==0.1.11 && \
    python3 -m uv pip install --upgrade --no-cache-dir \
        clu \
        "jax[cpu]>=0.2.16,!=0.3.2" \
        "flax>=0.4.1" \
        "jaxlib>=0.1.65" && \
    python3 -m uv pip install --no-cache-dir \
        accelerate \
        datasets \
        hf-doc-builder \
        huggingface-hub \
        Jinja2 \
        librosa \
        numpy==1.26.4 \
        scipy \
        tensorboard \
        transformers

CMD ["/bin/bash"]