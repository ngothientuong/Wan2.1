from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import logging
import time
import os
from threading import Thread
import numpy as np
import imageio_ffmpeg as ffmpeg
from wan import WanT2V
from wan.configs import WAN_CONFIGS, SIZE_CONFIGS
from wan.utils.utils import cache_video, str2bool

# Setup logging
logging.basicConfig(level=logging.INFO)
app = FastAPI()

# Detect GPUs
NUM_GPUS = torch.cuda.device_count()
DEVICE = f"cuda:{torch.cuda.current_device()}" if NUM_GPUS > 0 else "cpu"

# Model cache
MODEL_CACHE = {}

def load_model(task, ckpt_dir):
    if task not in MODEL_CACHE:
        logging.info(f"🔄 Loading {task} model on {DEVICE}...")
        config = WAN_CONFIGS[task]
        model = WanT2V(config=config, checkpoint_dir=ckpt_dir).to(DEVICE).half()

        # Enable TensorRT optimizations
        model = torch.compile(model, backend="tensorrt")

        MODEL_CACHE[task] = model
    return MODEL_CACHE[task]

class VideoRequest(BaseModel):
    task: str
    prompt: str
    size: str = "1280*720"  # ✅ Matches WAN 2.1 generate.py format
    num_frames: int = 320  # ✅ Default 20 seconds at 16 FPS
    fps: int = 16
    seed: int = 42
    ckpt_dir: str
    offload_model: bool = False
    t5_cpu: bool = False
    sample_steps: int = 4
    sample_shift: float = 3.0
    sample_guide_scale: float = 3.0
    save_file: str = None
    image: str = None  # Image-to-Video input support

def save_video_async(frames, output_file, fps):
    """Save video using FFmpeg asynchronously to prevent disk I/O bottlenecks."""
    writer = ffmpeg.write_frames(output_file, (1280, 720), fps=fps)
    writer.send(None)  # Initialize
    for frame in frames:
        writer.send(frame)
    writer.close()

def interpolate_frames(keyframes, interval):
    """Temporary fallback method for frame interpolation."""
    return [frame for frame in keyframes for _ in range(interval)]  # No Optical Flow yet

def generate_video(request: VideoRequest):
    """Optimized video generation using WAN 2.1 models."""
    start_time = time.time()
    torch.manual_seed(request.seed)

    model = load_model(request.task, request.ckpt_dir)

    # Compute keyframes at an interval of 4 for efficiency
    keyframe_interval = 4
    num_keyframes = request.num_frames // keyframe_interval
    batch_size = 32  # Adjustable based on available GPU memory
    num_batches = (num_keyframes + batch_size - 1) // batch_size

    # Generate keyframes in batches
    keyframes = []
    for i in range(num_batches):
        batch_start = i * batch_size
        batch_end = min((i + 1) * batch_size, num_keyframes)
        batch_prompts = [request.prompt] * (batch_end - batch_start)

        with torch.no_grad():
            batch_frames = model.generate(
                prompts=batch_prompts,
                num_frames=1,
                size=request.size,
                device=DEVICE
            )
        keyframes.extend(batch_frames)

    # Apply Keyframe Duplication (Temporary Fix)
    interpolated_frames = interpolate_frames(keyframes, keyframe_interval)

    # Save video asynchronously
    output_file = request.save_file if request.save_file else f"output_{int(time.time())}.mp4"
    Thread(target=save_video_async, args=(interpolated_frames, output_file, request.fps)).start()

    elapsed = time.time() - start_time
    logging.info(f"✅ Video generated in {elapsed:.2f} seconds")
    return {"output_path": output_file, "time_seconds": elapsed}

@app.post("/generate/")
async def generate_api(request: VideoRequest):
    """API Endpoint for video generation."""
    try:
        return generate_video(request)
    except Exception as e:
        logging.error(f"❌ Error generating video: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Initialize multi-GPU if available
if NUM_GPUS > 1:
    torch.distributed.init_process_group(backend="nccl")
