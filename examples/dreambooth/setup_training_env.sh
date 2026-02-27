#!/bin/bash

# Extend NCCL timeouts
export NCCL_SOCKET_TIMEOUT=7200000
export DEEPSPEED_TIMEOUT=7200000

# Set CPU threading optimizations
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb=512

# Increase system shared memory limits
sudo sysctl -w kernel.shmmax=85899345920
sudo sysctl -w kernel.shmall=2097152

# Enable NCCL debugging for diagnostics
export NCCL_DEBUG=INFO

# Optional: Set NCCL topology optimization 
# Uncomment if needed after checking nvidia-smi topo -m
# export NCCL_P2P_LEVEL=PHB

# Persist changes to sysctl
echo "kernel.shmmax=85899345920" | sudo tee -a /etc/sysctl.conf
echo "kernel.shmall=2097152" | sudo tee -a /etc/sysctl.conf
sudo sysctl -p