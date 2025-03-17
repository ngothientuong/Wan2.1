#!/bin/bash
set -euxo pipefail  # Enable error handling

# ========== 🛠️ 1️⃣ Update System & Install Core Packages ==========
echo "🔧 Updating system and installing required base packages..."
sudo apt-get update && sudo apt-get install -y \
    ca-certificates curl gnupg \
    git wget unzip jq tmux htop software-properties-common \
    libgl1-mesa-glx nano

# ========== 🔥 2️⃣ Install & Configure NVIDIA Drivers ==========
echo "🔥 Installing NVIDIA drivers if not already installed..."
if ! command -v nvidia-smi &> /dev/null; then
    sudo apt-get install -y nvidia-driver-550
    sudo reboot
fi

# Install CUDA & cuDNN to support containers (host does not need to run CUDA)
if ! dpkg -l | grep -q "cuda-toolkit"; then
    sudo apt-get install -y cuda-toolkit-12-2 libcudnn8 libcudnn8-dev
fi

# Ensure CUDA libraries are available inside containers
echo "🚀 Configuring CUDA environment..."
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# ========== 🐳 3️⃣ Install Docker & NVIDIA Container Toolkit ==========
echo "🐳 Installing Docker..."
if ! command -v docker &> /dev/null; then
    sudo apt-get install -y docker.io
    sudo usermod -aG docker ubuntu
    newgrp docker
fi

echo "🔧 Installing NVIDIA Container Toolkit..."
if ! command -v nvidia-container-runtime &> /dev/null; then
    sudo apt-get install -y nvidia-container-toolkit
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
fi

# ========== 🔹 4️⃣ Install Azure CLI (For ACR Authentication) ==========
echo "🔹 Installing Azure CLI..."
if ! command -v az &> /dev/null; then
    curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
fi

echo "🎉✅ Host is now fully supporting the container!"
