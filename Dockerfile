# Use the official TensorFlow 2.14.0 GPU-enabled Docker image with Python 3.10
FROM tensorflow/tensorflow:2.14.0-gpu

# Install necessary system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    curl \
    gnupg \
    ca-certificates \
    libgl1 \
    python3.10 \
    python3.10-dev \
    python3.10-distutils \
    && rm -rf /var/lib/apt/lists/*

# Install pip for Python 3.10
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.10 get-pip.py && \
    rm get-pip.py

# Upgrade pip
RUN pip install --upgrade pip

# Install virtualenv
RUN pip install virtualenv

# Create and activate virtual environment
RUN virtualenv -p python3.10 /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install PyTorch with CUDA 11.8 support
RUN pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 \
    -f https://download.pytorch.org/whl/torch_stable.html

# Install PyTorch Geometric and its dependencies automatically
RUN pip install torch-geometric -f https://data.pyg.org/whl/torch-2.0.1+cu118.html

# Copy project files
COPY . /data/NaroNet/

# Change ownership to non-root user (optional but recommended)
RUN chown -R 1000:1000 /data/NaroNet

# Set working directory
WORKDIR /data/NaroNet/

# Install project dependencies
RUN pip install -e .  # This will install NaroNet along with its dependencies, including numpy<2.0.0
RUN pip install imbalanced-learn==0.10.1  # Already specified in setup.py, but included here for redundancy

# Downgrade protobuf if necessary
RUN pip install protobuf==3.20.3

# Expose port 8888
EXPOSE 8888

# Set default command
CMD ["/bin/bash"]
