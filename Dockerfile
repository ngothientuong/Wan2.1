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

# ENV CUDA_VISIBLE_DEVICES=0,1  # Uncomment if limiting GPUs (default: use all)

# Install System Dependencies
RUN apt-get update && apt-get install -y \
  python3-pip git wget curl libgl1-mesa-glx ffmpeg \
  && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt and install Python Libraries
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Pre-download WAN 2.1 Model (Ensures Fast Boot Time)
RUN huggingface-cli download Wan-AI/Wan2.1-T2V-14B --local-dir ${MODEL_DIR} --revision main \
  && test -d ${MODEL_DIR} || exit 1  # Ensure model is downloaded

# Verify CUDA, Multi-GPU & PyTorch AMP Support
RUN python3 -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()} | GPUs: {torch.cuda.device_count()}')"
RUN python3 -c "import torch; print(f'Using BF16 AMP: {torch.backends.cuda.matmul.allow_tf32}')"

# Set Work Directory & Copy API Code
WORKDIR /app
COPY . /app

# Copy the startup script
COPY startup.sh /app/startup.sh
RUN chmod +x /app/startup.sh

# Expose API Port
EXPOSE 8000

# Run the startup script
CMD ["/app/startup.sh"]