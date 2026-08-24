FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python-is-python3 \
    git \
    wget \
    nano \
    build-essential \
    ffmpeg \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip

# PyTorch CUDA
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Python libraries
RUN pip install \
    numpy==1.26.4 \
    matplotlib \
    scikit-learn \
    opencv-python \
    tqdm \
    pillow \
    h5py \
    gymnasium \
    stable-baselines3

CMD ["bash"]
