#ARG BASE_IMAGE
#FROM ${BASE_IMAGE}
FROM https://nvcr.io/nvidia/pytorch:22.09-py3

ARG DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"
ENV TZ=US/San_Francisco
ENV MPI=openmpi
ENV PYTHONPATH=/opt/conda/bin/python
ENV CUDA_HOME=/usr/local/cuda-11.8
ENV HTTP_PROXY=http://19.12.80.237:83/

# install cuda nvcc
RUN conda config --set proxy_servers.http http://19.12.80.237:83/
RUN conda install -c nvidia/label/cuda-11.8.0 cuda-nvcc
# RUN conda install -c conda-forge gcc did not work
# RUN conda install -c conda-forge gcc==12.1.0
# RUN conda install gcc_linux-64
# RUN conda install -c conda-forge gxx_linux-64

# update git-lfs
RUN http_proxy=http://19.12.80.237:83/ apt-get update \
    && http_proxy=http://19.12.80.237:83/ apt-get install -y wget git-lfs ffmpeg
    
# install opencv
RUN  pip --proxy=19.12.80.237:83 install \
     opencv-python==4.5.2.54 \
     opencv-python-headless==4.5.2.54 \
     timm

# install imageio
RUN  pip --proxy=19.12.80.237:83 install \
     albumentations==0.4.3 \
     pudb==2019.2 \
     imageio==2.9.0 \
     imageio-ffmpeg==0.4.2 

# install pytorch lightning     
RUN  pip --proxy=19.12.80.237:83 install \
     pytorch-lightning \
     torchmetrics==0.6 \
     omegaconf==2.1.1 \
     test-tube>=0.7.5 

# install transformers     
RUN  pip --proxy=19.12.80.237:83 install \
     streamlit>=0.73.1 \
     einops==0.3.0 \
     transformers==4.19.2 \
     webdataset==0.2.5 

# install clip     
RUN  pip --proxy=19.12.80.237:83 install \
     open-clip-torch==2.7.0 \
     gradio==3.11 \
     kornia==0.6 \
     invisible-watermark>=0.1.5 
     
# install diffusers
RUN  pip --proxy=19.12.80.237:83 install \
     streamlit-drawable-canvas==0.8.0 \
     diffusers  

# install huggingface accelerated training
RUN  pip --proxy=19.12.80.237:83 install \
     accelerate \
     scipy \
     safetensors 

# install xformers required packages
RUN  pip --proxy=19.12.80.237:83 install \
     numpy \
     pyre-extensions==0.0.23 \
     # Requirements for flash_attn
     einops 

# compile stablediffusion
# COPY ./stablediffusion /stablediffusion
# WORKDIR /stablediffusion
# RUN pip install -r requirements.txt
# RUN pip install -e . 

# compile xformers
# WORKDIR /
# COPY ./xformers /xformers
# WORKDIR /xformers
# RUN pip install -r requirements.txt
# RUN pip install -v -e .

# WORKDIR /stablediffusion

