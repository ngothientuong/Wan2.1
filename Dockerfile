# Use NVIDIA PyTorch image with CUDA 12.4 and cuDNN 8.9 for best compatibility
FROM nvcr.io/nvidia/pytorch:24.02-py3

# Set environment variables for optimization
ENV DEBIAN_FRONTEND=noninteractive
ENV HF_HOME="/root/.cache/huggingface"
ENV MODEL_DIR="/models/Wan2.1-T2V-14B"
ENV PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
ENV CUDA_LAUNCH_BLOCKING=0
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV CUDA_HOME=/usr/local/cuda

# Install system dependencies
RUN apt-get update && apt-get install -y \
  ffmpeg libgl1-mesa-glx git wget curl \
  && rm -rf /var/lib/apt/lists/*

# Install base Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Install additional optimization dependencies
COPY additional-requirements.txt /app/additional-requirements.txt
RUN pip install --no-cache-dir -r /app/additional-requirements.txt

# Pre-download WAN 2.1 model to avoid runtime delays
RUN huggingface-cli download Wan-AI/Wan2.1-T2V-14B --local-dir ${MODEL_DIR} --revision main

# Set working directory and copy application files
WORKDIR /app
COPY . /app

# Expose FastAPI port
EXPOSE 8000

# Start FastAPI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
