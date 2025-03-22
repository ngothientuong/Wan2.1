#!/bin/bash
set -euxo pipefail

echo "🚀 Starting host setup for WAN2.1..."

# ========== 🛠️ 1️⃣ Update system and install essentials ==========
sudo apt-get update
sudo apt-get install -y \
  ca-certificates curl gnupg lsb-release \
  git wget unzip jq tmux htop nano \
  software-properties-common libgl1

# ========== ⚡ 2️⃣ Install NVIDIA driver + CUDA toolkit (host) ==========
# Your host image is already using R550 + CUDA 12.4, but this makes sure it's all installed

if ! command -v nvidia-smi &>/dev/null; then
  echo "🔧 Installing NVIDIA driver 550..."
  sudo apt-get install -y nvidia-driver-550
  sudo reboot
fi

# Install the CUDA toolkit 12.4 (for local compilation if needed — safe & compatible)
if ! dpkg -l | grep -q "cuda-toolkit-12-4"; then
  echo "📦 Installing CUDA toolkit 12.4..."
  sudo apt-get install -y cuda-toolkit-12-4
fi

# ========== 🐳 3️⃣ Install Docker & NVIDIA Container Toolkit ==========
if ! command -v docker &>/dev/null; then
  echo "🐳 Installing Docker..."
  sudo apt-get install -y docker.io
  sudo usermod -aG docker ubuntu
  newgrp docker
fi

echo "🧠 Checking NVIDIA Container Toolkit..."
if ! command -v nvidia-ctk &>/dev/null; then
  echo "🧠 Installing NVIDIA Container Toolkit..."
  sudo apt-get install -y nvidia-container-toolkit
  sudo nvidia-ctk runtime configure --runtime=docker
  sudo systemctl restart docker
else
  echo "✅ NVIDIA Container Toolkit already installed."
fi


# ========== 🔐 4️⃣ Install Azure CLI (for ACR login) ==========
if ! command -v az &>/dev/null; then
  echo "🔐 Installing Azure CLI..."
  curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
fi

echo "✅ user_data.sh completed: Host is ready for CUDA 12.5 containers."
