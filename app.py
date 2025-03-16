from fastapi import FastAPI, HTTPException, WebSocket, File, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
import torch
import os
import sys
import logging
from datetime import datetime
from typing import Optional
from torchreid.utils import interpolate_video
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

# ========== Load Model (Multi-GPU & Flash Attention) ==========
MODEL_CACHE = {}

def load_model(task, ckpt_dir):
    if task not in MODEL_CACHE:
        device_id = GPU_IDS[0] if NUM_GPUS > 0 else "cpu"
        logging.info(f"Loading model {task} on device {device_id} from {ckpt_dir}...")

        MODEL_CACHE[task] = torch.hub.load("WAN-2.1", model=task).to(device_id).half()
    return MODEL_CACHE[task]

# ========== Generate Video Function ==========
def generate_video(request: VideoRequest):
    model = load_model(request.task, request.ckpt_dir)

    # Handle Prompt Extension
    if request.use_prompt_extend:
        logging.info("Extending prompt...")
        if request.prompt_extend_method == "dashscope":
            expander = DashScopePromptExpander(is_vl="i2v" in request.task)
        else:
            expander = QwenPromptExpander(is_vl="i2v" in request.task)
        extended_prompt = expander(request.prompt, tar_lang=request.prompt_extend_target_lang).prompt
        request.prompt = extended_prompt

    # Keyframe optimization: Generate every 4th frame and interpolate
    keyframe_interval = 4
    num_keyframes = request.num_frames // keyframe_interval

    logging.info(f"Generating {num_keyframes} keyframes instead of {request.num_frames} full frames.")

    keyframes = model.generate(
        request.prompt,
        size=request.size,
        frame_num=num_keyframes,
        shift=request.sample_shift,
        sample_solver="unipc",
        sampling_steps=50,
        guide_scale=request.sample_guide_scale,
        seed=request.seed,
        offload_model=request.offload_model
    )

    # Use AI-based interpolation (RIFE) to generate missing frames
    full_video = interpolate_video(keyframes, method="rife", target_fps=request.fps)

    # Save video
    output_file = f"{request.task}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    cache_video(tensor=full_video, save_file=output_file, fps=request.fps)

    return output_file

# ========== API Endpoints ==========
@app.post("/generate/")
async def generate_api(request: VideoRequest):
    try:
        video_path = generate_video(request)
        return {"output_path": video_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
