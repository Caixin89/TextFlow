# Use PyTorch CUDA *devel* image so nvcc is available for building extensions
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda


WORKDIR /app

# Install a minimal set of commonly used apt packages without bloating the image
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    git \
    curl \
    wget \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir notebook
# Ensure ipykernel is registered for the environment used by the Jupyter server
RUN python -m ipykernel install --sys-prefix --name python3 --display-name "Python 3 (ipykernel)"

RUN pip install --upgrade jupyter ipywidgets

# Copy source code
# COPY src/ ./src
