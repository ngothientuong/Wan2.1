from fastapi import FastAPI, HTTPException, WebSocket, File, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
import torch
import os
import sys
import logging
import shutil
from datetime import datetime
from typing import Optional
from PIL import Image
import random

import wan
from wan.configs import WAN_CONFIGS, SIZE_CONFIGS, MAX_AREA_CONFIGS, SUPPORTED_SIZES
from wan.utils.utils import cache_video, cache_image
from wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander

# ========== Logging Setup ==========
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(stream=sys.stdout)]
)

# ========== Initialize FastAPI App ==========
app = FastAPI(title="WAN 2.1 Text-to-Video API")

# ========== Multi-GPU Handling ==========
NUM_GPUS = torch.cuda.device_count()
GPU_IDS = list(range(NUM_GPUS))

if NUM_GPUS > 0:
    logging.info(f"✅ Multi-GPU enabled: {NUM_GPUS} GPUs detected.")
else:
    logging.info("⚠️ No GPU detected! Running on CPU.")

# ========== Model Cache (Avoid Reloading on Every Request) ==========
MODEL_CACHE = {}

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
    image: Optional[str] = None  # Required for image-to-video tasks

# ========== Load Model (Multi-GPU & Caching) ==========
def load_model(task, ckpt_dir):
    if task not in MODEL_CACHE:
        device_id = GPU_IDS[0] if NUM_GPUS > 0 else "cpu"
        logging.info(f"Loading model {task} on device {device_id} from {ckpt_dir}...")

        MODEL_CACHE[task] = wan.WanT2V(
            config=WAN_CONFIGS[task],
            checkpoint_dir=ckpt_dir,
            device_id=device_id,
            torch_dtype=torch.float16 if NUM_GPUS > 0 else torch.float32
        )
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

    # Generate Video
    logging.info(f"Generating video with task: {request.task}, prompt: {request.prompt}")
    output = model.generate(
        request.prompt,
        size=SIZE_CONFIGS[request.size],
        frame_num=request.num_frames,
        shift=request.sample_shift,
        sample_solver="unipc",
        sampling_steps=50,
        guide_scale=request.sample_guide_scale,
        seed=request.seed,
        offload_model=request.offload_model
    )

    # Save Video (Server-Side)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{request.task}_{timestamp}.mp4"
    cache_video(tensor=output[None], save_file=output_file, fps=request.fps)

    return output_file

# ========== API Endpoints ==========
@app.get("/healthcheck")
def health_check():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return {"status": "running", "device": device, "num_gpus": NUM_GPUS}

@app.post("/generate/")
async def generate_api(request: VideoRequest):
    try:
        video_path = generate_video(request)
        return {"output_path": video_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download/{filename}")
async def download_file(filename: str):
    """Allows the client to download the generated file."""
    file_path = os.path.join(os.getcwd(), filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(file_path, media_type="video/mp4", filename=filename)

@app.post("/generate_and_download/")
async def generate_and_download(request: VideoRequest):
    """Generates video and streams it back to the client."""
    try:
        video_path = generate_video(request)

        def file_iterator():
            """Streams the file in chunks for efficient downloading."""
            with open(video_path, "rb") as video_file:
                while chunk := video_file.read(4096):
                    yield chunk

        return StreamingResponse(file_iterator(), media_type="video/mp4", headers={"Content-Disposition": f"attachment; filename={video_path}"})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ========== WebSocket for Real-Time Updates ==========
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            request = VideoRequest(**data)
            video_path = generate_video(request)
            await websocket.send_json({"status": "completed", "video_path": video_path})
    except Exception as e:
        await websocket.send_json({"error": str(e)})
    finally:
        await websocket.close()
