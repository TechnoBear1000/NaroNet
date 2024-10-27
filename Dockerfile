# Dockerfile for NaroNet

# Base image with CUDA support and Python 3.9
FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.9, python3.9-dev, and other necessary packages
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.9 python3.9-dev python3.9-distutils python3-pip python3-opencv && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Update alternatives to set Python 3.9 as the default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1

# Install pip for Python 3.9
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.9

# Create and activate NaroNet directory
WORKDIR /naronet

# Copy all required files into the container
COPY . .

# Install `numpy` first to avoid compatibility issues
RUN python3.9 -m pip install --upgrade numpy

# Install Python dependencies, including pytorch-geometric
RUN python3.9 -m pip install \
    torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113 && \
    python3.9 -m pip install \
    torch-scatter -f https://data.pyg.org/whl/torch-1.10.0+cu113.html && \
    python3.9 -m pip install \
    torch-sparse -f https://data.pyg.org/whl/torch-1.10.0+cu113.html && \
    python3.9 -m pip install \
    torch-cluster -f https://data.pyg.org/whl/torch-1.10.0+cu113.html && \
    python3.9 -m pip install \
    torch-spline-conv -f https://data.pyg.org/whl/torch-1.10.0+cu113.html && \
    python3.9 -m pip install \
    torch-geometric && \
    python3.9 -m pip install \
    hyperopt \
    xlsxwriter \
    matplotlib \
    seaborn \
    imgaug \
    tensorboard \
    openTSNE \
    openpyxl

# Set command to run the application (this command can be overridden if needed)
CMD ["python3.9", "src/main.py"]