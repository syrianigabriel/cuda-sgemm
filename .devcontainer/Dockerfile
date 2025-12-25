FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

RUN apt-get update && \
    apt-get install -y git && \
    rm -rf /var/lib/apt/lists/*

RUN useradd -m -s /bin/bash cudauser \
    && echo "cudauser ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers