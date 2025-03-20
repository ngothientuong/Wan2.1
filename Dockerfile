# ✅ Best NVIDIA image for PyTorch 2.4 + CUDA 12.3
FROM nvcr.io/nvidia/pytorch:24.06-py3

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

# Ensure the environment uses CUDA 12.3 (Matches FlashAttention wheel)
ENV CUDA_HOME=/usr/local/cuda-12.3
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# Install System Dependencies (CUDA + Build Tools + Python 3.10)
RUN apt-get update && apt-get install -y \
  python3.10 python3.10-venv python3.10-dev python3-pip \
  git wget curl ffmpeg libgl1-mesa-glx \
  ninja-build build-essential cmake unzip \
  && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as Default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
  update-alternatives --config python3

# Verify Python version
RUN python3 --version

# Upgrade pip and install core dependencies
RUN pip install --upgrade pip setuptools wheel packaging

# Debug CUDA version inside the container
RUN echo "🔍 Checking CUDA version inside container..." && \
  nvcc --version && \
  python -c "import torch; print('Torch CUDA Version:', torch.version.cuda)"

# Copy Requirements Files
COPY requirements.txt /app/requirements.txt

# Install Python Dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt

# Uninstall any preinstalled flash_attn (ensures a clean installation)
RUN pip uninstall -y flash-attn

# ✅ Install the correct FlashAttention wheel (CUDA 12.3, PyTorch 2.4)
RUN pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# Clone RAFT Repository
RUN git clone https://github.com/princeton-vl/RAFT.git /app/RAFT

# ✅ Fix: Download and Extract RAFT Models
WORKDIR /app/RAFT
RUN ./download_models.sh && \
  mv models/* /app/RAFT/core/ && \
  rm -rf models models.zip

# ✅ Fix: Ensure models directory exists and pre-download WAN 2.1 Model
RUN mkdir -p ${MODEL_DIR} && \
  huggingface-cli download Wan-AI/Wan2.1-T2V-14B --local-dir ${MODEL_DIR} --revision main

# Set Working Directory & Copy Application Files
WORKDIR /app
COPY . /app

# ✅ Fix: Ensure `models` symlink is set correctly inside the container
RUN ln -s /models /app/models

# Expose API Port
EXPOSE 8000

# Run Application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
