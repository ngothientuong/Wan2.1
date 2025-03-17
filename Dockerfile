# Use NVIDIA PyTorch base image with CUDA 12.3 and cuDNN 8.9
FROM nvcr.io/nvidia/pytorch:24.01-py3

# Set environment variables for optimization
ENV DEBIAN_FRONTEND=noninteractive
ENV HF_HOME="/root/.cache/huggingface"
ENV MODEL_DIR="/models/Wan2.1-T2V-14B"
ENV PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
ENV CUDA_LAUNCH_BLOCKING=0

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

# Set working directory and copy application code
WORKDIR /app
COPY . /app

# Expose FastAPI port
EXPOSE 8000

# Start FastAPI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
