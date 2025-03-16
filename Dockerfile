# Use NVIDIA PyTorch Base Image (CUDA-Optimized)
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Set Environment Variables
ENV DEBIAN_FRONTEND=noninteractive
ENV HF_HOME="/root/.cache/huggingface"
ENV MODEL_DIR="/models/Wan2.1-T2V-14B"
ENV TORCH_HOME="/root/.cache/torch"

# Install System Dependencies & Python Libraries in One Step
RUN apt-get update && apt-get install -y \
  python3-pip git wget curl libgl1-mesa-glx ffmpeg \
  && rm -rf /var/lib/apt/lists/* \
  && pip install --no-cache-dir torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu121 \
  && pip install --no-cache-dir fastapi uvicorn pydantic \
  transformers diffusers huggingface_hub pillow tqdm ninja pyyaml \
  easydict ftfy einops imageio dashscope torchreid numpy \
  && pip install --no-cache-dir packaging \
  && pip install --no-cache-dir flash-attn --no-build-isolation \
  && pip install --no-cache-dir scipy opencv-python-headless gdown \
  && pip install --no-cache-dir tensorboard tensorboardX protobuf cython \
  && pip install --no-cache-dir imageio[ffmpeg] huggingface_hub \
  && huggingface-cli download Wan-AI/Wan2.1-T2V-14B --local-dir ${MODEL_DIR} --revision main \
  && test -d ${MODEL_DIR} || exit 1  # Ensure model is downloaded

# Set Work Directory & Copy API Code
WORKDIR /app
COPY . /app

# Expose API Port
EXPOSE 8000

# Run FastAPI Server with Improved Stability
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--timeout-keep-alive", "120"]