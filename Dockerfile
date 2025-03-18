# Use NVIDIA’s Latest PyTorch Image (Ensure CUDA version matches your system)
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

# Install System Dependencies (Ensure CUDA Development Tools are Installed)
RUN apt-get update && apt-get install -y \
  python3-pip git wget curl ffmpeg libgl1-mesa-glx \
  ninja-build build-essential cmake \
  && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install core dependencies
RUN pip install --upgrade pip setuptools wheel packaging

# Copy Requirements Files
COPY requirements.txt /app/requirements.txt

# Install Python Dependencies (Using Cached Layers)
RUN pip install --no-cache-dir -r /app/requirements.txt

# Uninstall any preinstalled flash_attn (ensures a clean installation)
RUN pip uninstall -y flash-attn

# Install flash_attn from the correct prebuilt wheel
RUN pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu124torch2.4-cp310-cp310-linux_x86_64.whl

# Clone & Install RAFT from Source
RUN pip install --no-deps git+https://github.com/princeton-vl/RAFT.git

# Pre-download WAN 2.1 Model
RUN huggingface-cli download Wan-AI/Wan2.1-T2V-14B --local-dir ${MODEL_DIR} --revision main

# Set Working Directory & Copy Application Files
WORKDIR /app
COPY . /app

# Expose API Port
EXPOSE 8000

# Run Application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
