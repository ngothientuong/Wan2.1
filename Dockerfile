# Use NVIDIA’s Latest PyTorch Image (CUDA 12.3 + CuDNN 8.9)
FROM nvcr.io/nvidia/pytorch:24.01-py3

# Set Environment Variables for Performance Optimization
ENV DEBIAN_FRONTEND=noninteractive
ENV HF_HOME="/root/.cache/huggingface"
ENV MODEL_DIR="/models/Wan2.1-T2V-14B"
ENV TORCH_HOME="/root/.cache/torch"
ENV PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
ENV CUDA_LAUNCH_BLOCKING=1
# ENV CUDA_VISIBLE_DEVICES=0,1 - Only set if wanting exactly just 2 and no more GPUs even with machine with higher number of GPUs! Comment out to use all GPUs

# Install System Dependencies & Python Libraries
RUN apt-get update && apt-get install -y \
  python3-pip git wget curl libgl1-mesa-glx ffmpeg \
  && rm -rf /var/lib/apt/lists/* \
  \
  # Install Required PyTorch Libraries & Dependencies
  && pip install --no-cache-dir torch torchvision torchaudio \
  && pip install --no-cache-dir fastapi uvicorn pydantic \
  transformers diffusers huggingface_hub pillow tqdm ninja pyyaml \
  easydict ftfy einops imageio dashscope torchreid numpy \
  && pip install --no-cache-dir packaging flash-attn --no-build-isolation \
  && pip install --no-cache-dir scipy opencv-python-headless gdown \
  && pip install --no-cache-dir tensorboard tensorboardX protobuf cython \
  && pip install --no-cache-dir imageio[ffmpeg] huggingface_hub accelerate \
  \
  # Install Performance Optimizations (Memory & Speed)
  && pip install --no-cache-dir xformers triton deepspeed \
  && pip install --no-cache-dir nvidia-pyindex nvidia-tensorrt \
  && pip install --no-cache-dir bitsandbytes \
  && pip install --no-cache-dir py3nvml \
  && pip install --no-cache-dir fastertransformer \
  \
  # Enable PyTorch Native Compilation & Flash Attention for Speed
  && pip install --no-cache-dir torch-sdp-attn \
  && pip install --no-cache-dir torch-compile \
  \
  # Download WAN 2.1 Model (Ensures It Exists & is Cached)
  && huggingface-cli download Wan-AI/Wan2.1-T2V-14B --local-dir ${MODEL_DIR} --revision main \
  && test -d ${MODEL_DIR} || exit 1  # Ensure model is downloaded

# Verify CUDA, Multi-GPU & PyTorch AMP Support
RUN python3 -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()} | GPUs: {torch.cuda.device_count()}')"
RUN python3 -c "import torch; print(f'Using BF16 AMP: {torch.backends.cuda.matmul.allow_tf32}')"

# Set Work Directory & Copy API Code
WORKDIR /app
COPY . /app

# Expose API Port
EXPOSE 8000

# Run FastAPI Server with Stable Multi-GPU Support
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--timeout-keep-alive", "120", "--log-level", "debug"]
