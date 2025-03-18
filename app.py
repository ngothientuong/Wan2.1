from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator, root_validator
import torch
import torch.nn as nn
import logging
import time
import os
from threading import Thread
import cv2
import numpy as np
import imageio_ffmpeg as ffmpeg
from wan import WanT2V
from wan.configs import WAN_CONFIGS, SIZE_CONFIGS
import asyncio

try:
    from raft import RAFT
    from raft.utils.utils import InputPadder
    raft_model = RAFT()
    raft_model.load_model("/app/raft-things.pth")  # Ensure the model file exists
    USE_RAFT = True
    logging.info("✅ RAFT model loaded successfully.")
except Exception as e:
    USE_RAFT = False
    logging.warning(f"⚠️ RAFT not available, falling back to OpenCV optical flow: {e}")

# Configure logging
logging.basicConfig(level=logging.INFO)
app = FastAPI()

# Device configuration
NUM_GPUS = torch.cuda.device_count()
DEVICE = "cuda" if NUM_GPUS > 0 else "cpu"

# Model cache to avoid reloading
MODEL_CACHE = {}

def get_optimal_batch_size():
    """Dynamically adjust batch size based on available GPU memory."""
    total_memory = torch.cuda.get_device_properties(0).total_memory
    if total_memory > 40e9:  # A100 80GB
        return 128  # Large batch size
    elif total_memory > 20e9:  # A40, RTX 3090
        return 64
    else:
        return 32  # Safe default

def load_model(task, ckpt_dir, ulysses_size=1, ring_size=1, t5_fsdp=False, dit_fsdp=False):
    """Load or retrieve a cached WanT2V model."""
    key = (task, ulysses_size, ring_size, t5_fsdp, dit_fsdp)
    if key not in MODEL_CACHE:
        logging.info(f"🔄 Loading {task} model on {DEVICE}...")
        config = WAN_CONFIGS[task]
        model = WanT2V(config=config, checkpoint_dir=ckpt_dir, ulysses_size=ulysses_size, ring_size=ring_size).to(DEVICE).half()
        model = torch.compile(model, backend="tensorrt")

        if NUM_GPUS > 1:
            model = nn.DataParallel(model)  # Multi-GPU support without DDP complexity

        MODEL_CACHE[key] = model
    return MODEL_CACHE[key]

class VideoRequest(BaseModel):
    """Request model with validation and task-specific defaults."""
    task: str
    prompt: str
    size: str = "1280*720"
    num_frames: int = 320
    fps: int = 16
    seed: int = 42
    ckpt_dir: str
    offload_model: bool = False
    t5_cpu: bool = False
    t5_fsdp: bool = False
    dit_fsdp: bool = False
    ulysses_size: int = 1
    ring_size: int = 1
    sample_solver: str = "unipc"
    sample_steps: int = None  # Set dynamically based on task
    sample_shift: float = 3.0
    sample_guide_scale: float = 3.0
    save_file: str = None
    image: str = None  # For image-to-video tasks
    use_prompt_extend: bool = True
    prompt_extend_method: str = "local_qwen"
    prompt_extend_model: str = None
    prompt_extend_target_lang: str = "en"
    keyframe_interval: int = 30  # Allow keyframe interval customization

    @validator('task')
    def task_must_be_valid(cls, v):
        if v not in WAN_CONFIGS:
            raise ValueError(f"Invalid task: {v}")
        return v

    @validator('size')
    def size_must_be_valid(cls, v):
        if v not in SIZE_CONFIGS:
            raise ValueError(f"Invalid size: {v}")
        return v

    @validator('num_frames')
    def num_frames_must_be_4n_plus_1(cls, v):
        if (v - 1) % 4 != 0:
            raise ValueError("num_frames must be of the form 4n+1")
        return v

    @root_validator(pre=False)
    def set_sample_steps_default(cls, values):
        """Set sample_steps based on task if not provided."""
        if values.get('sample_steps') is None:
            task = values.get('task')
            if task.startswith('t2v'):
                values['sample_steps'] = 50
            elif task.startswith('i2v'):
                values['sample_steps'] = 40
        return values

def save_video_async(frames, output_file, fps):
    """Save video frames asynchronously."""
    writer = ffmpeg.write_frames(output_file, (1280, 720), fps=fps)
    writer.send(None)
    for frame in frames:
        writer.send(frame)
    writer.close()

def interpolate_frames(keyframes, interval):
    """Interpolate frames between keyframes using RAFT (if available) or OpenCV."""
    if USE_RAFT:
        logging.info("🚀 Using RAFT for GPU-accelerated frame interpolation.")
        return raft_interpolation(keyframes, interval)
    else:
        logging.info("⚠️ RAFT not available, using OpenCV optical flow instead.")
        return opencv_interpolation(keyframes, interval)

def raft_interpolation(keyframes, interval):
    """Interpolate frames using RAFT (GPU-accelerated optical flow)."""
    interpolated_frames = []
    for i in range(len(keyframes) - 1):
        frame1 = torch.tensor(keyframes[i]).permute(2, 0, 1).unsqueeze(0).float().cuda()
        frame2 = torch.tensor(keyframes[i + 1]).permute(2, 0, 1).unsqueeze(0).float().cuda()

        padder = InputPadder(frame1.shape)
        frame1, frame2 = padder.pad(frame1, frame2)

        with torch.no_grad():
            flow = raft_model(frame1, frame2, iters=20, test_mode=True)[0]

        for j in range(1, interval):
            alpha = j / interval
            interp_frame = frame1 + alpha * flow
            interp_frame = interp_frame.cpu().squeeze(0).permute(1, 2, 0).numpy()
            interpolated_frames.append(interp_frame)
    return interpolated_frames

def opencv_interpolation(keyframes, interval):
    """Fallback to OpenCV's Farneback method for optical flow."""
    interpolated_frames = []
    for i in range(len(keyframes) - 1):
        frame1 = keyframes[i]
        frame2 = keyframes[i + 1]
        flow = cv2.calcOpticalFlowFarneback(cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY),
                                            cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY),
                                            None, 0.5, 3, 15, 3, 5, 1.2, 0)
        for j in range(interval):
            alpha = j / interval
            interp_frame = cv2.addWeighted(frame1, 1 - alpha, frame2, alpha, 0)
            interpolated_frames.append(interp_frame)
    return interpolated_frames

def generate_video(request: VideoRequest):
    """Generate video based on the request parameters."""
    start_time = time.time()
    torch.manual_seed(request.seed)

    model = load_model(request.task, request.ckpt_dir, request.ulysses_size, request.ring_size, request.t5_fsdp, request.dit_fsdp)

    keyframe_interval = request.keyframe_interval
    num_keyframes = request.num_frames // keyframe_interval
    batch_size = get_optimal_batch_size()
    num_batches = (num_keyframes + batch_size - 1) // batch_size

    keyframes = []
    for i in range(num_batches):
        batch_prompts = [request.prompt] * min(batch_size, num_keyframes - i * batch_size)
        with torch.no_grad():
            batch_frames = model.generate(prompts=batch_prompts, num_frames=1, size=request.size, device=DEVICE)
        keyframes.extend(batch_frames)

    interpolated_frames = interpolate_frames(keyframes, keyframe_interval)
    output_file = request.save_file or f"output_{int(time.time())}.mp4"
    Thread(target=save_video_async, args=(interpolated_frames, output_file, request.fps)).start()

    return {"output_path": output_file, "time_seconds": time.time() - start_time}
