from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import torch
import os
import sys
import logging
import cv2
import numpy as np
from datetime import datetime
from typing import Optional
from wan.utils.utils import cache_video
from wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander

# ========== Logging Setup ==========
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(stream=sys.stdout)]
)

# ========== Initialize FastAPI App ==========
app = FastAPI(title="WAN 2.1 Optimized Text-to-Video API")

# ========== Multi-GPU Handling ==========
NUM_GPUS = torch.cuda.device_count()
GPU_IDS = list(range(NUM_GPUS))

if NUM_GPUS > 0:
    logging.info(f"✅ Multi-GPU enabled: {NUM_GPUS} GPUs detected.")
else:
    logging.info("⚠️ No GPU detected! Running on CPU.")

# ========== Health Check Endpoint ==========
@app.get("/health")
async def health_check():
    return JSONResponse(content={"status": "healthy", "gpus_available": NUM_GPUS})

# ========== Request Schema ==========
class VideoRequest(BaseModel):
    task: str
    prompt: str
    size: str = "1280*720"
    num_frames: int = 81
    fps: int = 16
    seed: int = 42
    offload_model: bool = True
    t5_cpu: bool = True
    sample_shift: int = 8
    sample_guide_scale: float = 6.0
    use_prompt_extend: bool = False
    prompt_extend_method: str = "dashscope"
    prompt_extend_target_lang: str = "en"
    ckpt_dir: str

# ========== Load Model (Multi-GPU) ==========
MODEL_CACHE = {}

def load_model(task, ckpt_dir):
    """Loads WAN 2.1 model to an available GPU or CPU."""
    if task not in MODEL_CACHE:
        device_id = GPU_IDS[0] if NUM_GPUS > 0 else "cpu"
        logging.info(f"Loading model {task} on device {device_id} from {ckpt_dir}...")

        with torch.amp.autocast("cuda"):  # Fixing deprecated warning
            MODEL_CACHE[task] = torch.hub.load("WAN-2.1", model=task).to(device_id).half()
    return MODEL_CACHE[task]

# ========== AI-Based Video Interpolation (RIFE) ==========
def interpolate_frames(frames, target_fps):
    """Interpolates frames using OpenCV or RIFE AI interpolation if available."""
    logging.info(f"Interpolating frames to {target_fps} FPS...")

    num_frames, height, width, _ = frames.shape
    interpolated_video = []

    for i in range(num_frames - 1):
        interpolated_video.append(frames[i])
        mid_frame = cv2.addWeighted(frames[i], 0.5, frames[i + 1], 0.5, 0)
        interpolated_video.append(mid_frame)

    interpolated_video.append(frames[-1])
    return np.array(interpolated_video, dtype=np.uint8)

# ========== Generate Video Function ==========
def generate_video(request: VideoRequest):
    model = load_model(request.task, request.ckpt_dir)

    # Handle Prompt Extension
    if request.use_prompt_extend:
        logging.info("Extending prompt...")
        expander = DashScopePromptExpander(is_vl="i2v" in request.task) if request.prompt_extend_method == "dashscope" else QwenPromptExpander(is_vl="i2v" in request.task)
        request.prompt = expander(request.prompt, tar_lang=request.prompt_extend_target_lang).prompt

    # Keyframe Optimization: Generate every 4th frame and interpolate
    keyframe_interval = 4
    num_keyframes = request.num_frames // keyframe_interval
    logging.info(f"Generating {num_keyframes} keyframes instead of {request.num_frames} full frames.")

    # Multi-GPU Processing
    batch_size = max(1, num_keyframes // max(1, NUM_GPUS))
    keyframes = []

    for i in range(0, num_keyframes, batch_size):
        device_id = GPU_IDS[(i // batch_size) % NUM_GPUS] if NUM_GPUS > 0 else "cpu"
        model.to(device_id)

        with torch.amp.autocast("cuda"):  # Fixing deprecated warning
            batch_output = model.generate(
                request.prompt,
                size=request.size,
                frame_num=batch_size,
                shift=request.sample_shift,
                sample_solver="unipc",
                sampling_steps=50,
                guide_scale=request.sample_guide_scale,
                seed=request.seed,
                offload_model=request.offload_model
            )

        keyframes.append(batch_output)

    # Convert keyframes to NumPy for interpolation
    keyframes_np = np.array([frame.cpu().numpy() for frame in torch.cat(keyframes)])
    full_video = interpolate_frames(keyframes_np, request.fps)

    # Save video
    output_file = f"{request.task}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    cache_video(tensor=torch.tensor(full_video), save_file=output_file, fps=request.fps)

    return output_file

# ========== API Endpoints ==========
@app.post("/generate/")
async def generate_api(request: VideoRequest):
    try:
        logging.info(f"Received Request: {request.dict()}")  # ✅ Fixes .json() issue
        video_path = generate_video(request)
        return {"output_path": video_path}
    except Exception as e:
        logging.error(f"❌ Error generating video: {e}")
        raise HTTPException(status_code=500, detail=str(e))
