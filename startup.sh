#!/bin/bash

# filepath: /app/startup.sh
set -e  # Exit immediately if a command fails

# Default number of workers based on GPU count
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
NUM_WORKERS=$((NUM_GPUS > 0 ? NUM_GPUS : 1))

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        *)
            echo "❌ Unknown argument: $1"
            exit 1
            ;;
    esac
done

echo "🖥️ Using $NUM_WORKERS Uvicorn workers..."

# Function to check and install missing packages
install_missing_packages() {
    echo "🔍 Checking for missing Python packages..."
    required_packages=(
        "uvicorn" "fastapi" "torch" "diffusers" "transformers" "accelerate" "numpy"
    )
    missing_packages=()

    for package in "${required_packages[@]}"; do
        python3 -c "import $package" 2>/dev/null || missing_packages+=("$package")
    done

    if [ ${#missing_packages[@]} -ne 0 ]; then
        echo "⚠️ Missing packages detected: ${missing_packages[@]}"
        echo "📦 Installing missing dependencies..."
        pip install --no-cache-dir "${missing_packages[@]}"
    else
        echo "✅ All required packages are installed."
    fi
}

# Retry logic for Uvicorn (Max 5 Attempts)
MAX_RETRIES=5
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    echo "🚀 Starting Uvicorn with $NUM_WORKERS workers..."
    if uvicorn app:app --host 0.0.0.0 --port 8000 --workers $NUM_WORKERS --timeout-keep-alive 120 --log-level debug; then
        break  # Success, exit loop
    else
        echo "❌ Uvicorn failed to start. Checking for missing packages..."
        install_missing_packages
        RETRY_COUNT=$((RETRY_COUNT + 1))
        echo "🔄 Retrying ($RETRY_COUNT/$MAX_RETRIES)..."
        sleep 5
    fi
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo "🚨 Uvicorn failed after $MAX_RETRIES attempts. Exiting."
    exit 1
fi
