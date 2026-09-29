FROM ubuntu:20.04
LABEL maintainer="Hugging Face"
LABEL repository="diffusers"

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get -y update \
    && apt-get install -y software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa

RUN apt install -y bash \
                   build-essential \
                   wget \
                   curl \
                   git \
                   git-lfs \
                   curl \
                   ca-certificates \
                   libsndfile1-dev \
                   python3.10 \
                   python3-pip \
                   python3.10-venv && \
    rm -rf /var/lib/apt/lists

# make sure to use venv
RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# download and install cann toolkit
RUN wget -q https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/Milan-ASL/Milan-ASL%20V100R001C18B800TP015/Ascend-cann-toolkit_8.0.RC2.alpha001_linux-"$(uname -i)".run && \
    chmod +x Ascend-cann-toolkit_8.0.RC2.alpha001_linux-"$(uname -i)".run && \
    ./Ascend-cann-toolkit_8.0.RC2.alpha001_linux-"$(uname -i)".run --quiet --install && \
    rm -f ./Ascend-cann-toolkit_8.0.RC2.alpha001_linux-"$(uname -i)".run

# download and install cann kernel
RUN wget -q https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/Milan-ASL/Milan-ASL%20V100R001C18B800TP015/Ascend-cann-kernels-910b_8.0.RC2.alpha001_linux.run && \
    chmod +x Ascend-cann-kernels-910b_8.0.RC2.alpha001_linux.run && \
    ./Ascend-cann-kernels-910b_8.0.RC2.alpha001_linux.run --quiet --install && \
    rm -f ./Ascend-cann-kernels-910b_8.0.RC2.alpha001_linux.run

# User must join group HwHiAiUser(1000) to use npu.
RUN groupadd -g 1000 HwHiAiUser && \
    usermod -a -G HwHiAiUser root

# set env variables to make npu available
ENV LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH:/usr/lib64
ENV ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest
ENV LD_LIBRARY_PATH=${ASCEND_TOOLKIT_HOME}/lib64:${ASCEND_TOOLKIT_HOME}/lib64/plugin/opskernel:${ASCEND_TOOLKIT_HOME}/lib64/plugin/nnengine:${ASCEND_TOOLKIT_HOME}/opp/built-in/op_impl/ai_core/tbe/op_tiling:$LD_LIBRARY_PATH
ENV PYTHONPATH=${ASCEND_TOOLKIT_HOME}/python/site-packages:${ASCEND_TOOLKIT_HOME}/opp/built-in/op_impl/ai_core/tbe:$PYTHONPATH
ENV PATH=${ASCEND_TOOLKIT_HOME}/bin:${ASCEND_TOOLKIT_HOME}/compiler/ccec_compiler/bin:$PATH
ENV ASCEND_AICPU_PATH=${ASCEND_TOOLKIT_HOME}
ENV ASCEND_OPP_PATH=${ASCEND_TOOLKIT_HOME}/opp
ENV TOOLCHAIN_HOME=${ASCEND_TOOLKIT_HOME}/toolkit
ENV ASCEND_HOME_PATH=${ASCEND_TOOLKIT_HOME}

# install Rust toolchain (for uv installation)
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
SHELL [ "/bin/bash", "-c", "source ~/.cargo/env"]

# pre-install the heavy dependencies (these can later be overridden by the deps from setup.py)
# TODO: update torch version after torch-npu updated
RUN python3.10 -m pip install --no-cache-dir --upgrade pip uv==0.1.11 && \
    python3.10 -m uv pip install --no-cache-dir \
        torch==2.2.0 \
        torch_npu==2.2.0 \
        torchvision \
        torchaudio \
        invisible_watermark && \
    python3.10 -m pip install --no-cache-dir \
        accelerate \
        datasets \
        hf-doc-builder \
        huggingface-hub \
        Jinja2 \
        librosa \
        numpy \
        scipy \
        tensorboard \
        transformers \
        matplotlib \
        sympy \
        decorator \
        attrs \
        requests

CMD ["/bin/bash"]
