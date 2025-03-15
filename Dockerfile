# Use NVIDIA PyTorch Base Image (CUDA-Optimized)
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Set Environment Variables
ENV DEBIAN_FRONTEND=noninteractive
ENV HF_HOME="/root/.cache/huggingface"
ENV MODEL_DIR="/models/Wan2.1-T2V-14B"

# Install System Dependencies & Python Libraries in One Step
RUN apt-get update && apt-get install -y \
  python3-pip git wget curl libgl1-mesa-glx \
  && rm -rf /var/lib/apt/lists/* \
  && pip install --no-cache-dir torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu121 \
  && pip install --no-cache-dir fastapi uvicorn pydantic \
  transformers diffusers huggingface_hub pillow \
  easydict ftfy einops imageio dashscope \
  && pip install huggingface_hub \
  && huggingface-cli download Wan-AI/Wan2.1-T2V-14B --local-dir ${MODEL_DIR} --revision main --depth 1

# Set Work Directory & Copy API Code
WORKDIR /app
COPY . /app

# Expose API Port
EXPOSE 8000

# Run FastAPI Server with Improved Stability
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--timeout-keep-alive", "120"]
