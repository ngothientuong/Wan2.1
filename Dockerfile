# Use NVIDIA’s Latest PyTorch Image (CUDA 12.3 + CuDNN 8.9)
FROM nvcr.io/nvidia/pytorch:24.01-py3

# Set Environment Variables for Performance Optimization
ENV DEBIAN_FRONTEND=noninteractive
ENV HF_HOME="/root/.cache/huggingface"
ENV MODEL_DIR="/models/Wan2.1-T2V-14B"
ENV TORCH_HOME="/root/.cache/torch"
ENV PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
ENV XFORMERS_FORCE_DISABLE_TRITON=1
ENV TF32_MATMUL=1
ENV CUDA_LAUNCH_BLOCKING=0
ENV PYTHONUNBUFFERED=1

# Install only essential system dependencies
RUN apt-get update && apt-get install -y \
  python3-pip git wget curl ffmpeg libgl1-mesa-glx \
  && rm -rf /var/lib/apt/lists/*

# Install base dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Install additional optimization dependencies
COPY additional-requirements.txt /app/additional-requirements.txt
RUN pip install --no-cache-dir -r /app/additional-requirements.txt

# Pre-download WAN 2.1 Model
RUN huggingface-cli download Wan-AI/Wan2.1-T2V-14B --local-dir ${MODEL_DIR} --revision main

WORKDIR /app
COPY . /app

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
