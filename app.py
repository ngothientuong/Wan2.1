from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator, root_validator, model_validator
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import logging
import time
import os
import sys
from threading import Thread
import cv2
import numpy as np
import imageio_ffmpeg as ffmpeg
from wan import WanT2V
from wan.configs import WAN_CONFIGS, SIZE_CONFIGS
import asyncio
import argparse
import subprocess
from huggingface_hub import snapshot_download  # Added for model downloading

# Model cache to avoid reloading
MODEL_CACHE = {}
# Set the root directory for model storage
MODEL_ROOT = "/mnt/storage/models"  # Mounted persistent volume
MODEL_WAN_T2V_14B = "Wan2.1-T2V-14B"
os.makedirs(MODEL_ROOT, exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
app = FastAPI()

# RAFT Model Loading
USE_RAFT = False
try:
    RAFT_PATH = "/app/RAFT/core"
    sys.path.append(RAFT_PATH)

    from raft import RAFT
    from utils.utils import InputPadder

    # Create RAFT model with dummy arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--small", action="store_true")
    parser.add_argument("--mixed_precision", action="store_true")
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--alternate_corr", action="store_true")
    args = parser.parse_args([])

    raft_model = RAFT(args)
    raft_model.to("cuda" if torch.cuda.is_available() else "cpu")

    # Load RAFT weights
    model_path = os.path.join(RAFT_PATH, "raft-things.pth")
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location="cuda" if torch.cuda.is_available() else "cpu")

        # Remove "module." prefix if present in state_dict keys
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        raft_model.load_state_dict(new_state_dict, strict=False)  # Load state dict
        raft_model.eval()

        USE_RAFT = True
        logging.info("✅ RAFT model loaded successfully.")
    else:
        logging.warning(f"⚠️ RAFT model file not found at {model_path}")
except Exception as e:
    logging.warning(f"⚠️ RAFT not available, falling back to OpenCV optical flow: {e}")

# Device configuration
NUM_GPUS = torch.cuda.device_count()
DEVICE = "cuda" if NUM_GPUS > 0 else "cpu"


def init_distributed():
    """Initialize the distributed environment for DDP."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        torch.distributed.init_process_group(backend='nccl', rank=rank, world_size=world_size)
        torch.cuda.set_device(rank % torch.cuda.device_count())
        logging.info(f"Initialized distributed process group for rank {rank}/{world_size}")
    else:
        logging.warning("Distributed environment variables not set. Running in single-process mode.")

def load_model(task, ckpt_dir=None, huggingface_repo_id=None, t5_fsdp=False, dit_fsdp=False):
    """Load or retrieve a cached WanT2V model utilizing multiple GPUs efficiently with DDP."""
    key = (task, t5_fsdp, dit_fsdp)
    if key not in MODEL_CACHE:
        logging.info(f"🔄 Loading {task} model for distributed setup...")

        # Only rank 0 downloads the model to avoid race conditions
        if not dist.is_initialized() or dist.get_rank() == 0:
            if not os.path.exists(ckpt_dir) or not os.listdir(ckpt_dir):
                logging.info(f"📥 Downloading model from Hugging Face to {ckpt_dir}...")
                os.makedirs(ckpt_dir, exist_ok=True)
                snapshot_download(repo_id=huggingface_repo_id, cache_dir=ckpt_dir, local_dir_use_symlinks=False)
        if dist.is_initialized():
            dist.barrier()  # Synchronize all processes after rank 0 downloads

        config = WAN_CONFIGS[task]
        model = WanT2V(config=config, checkpoint_dir=ckpt_dir)

        # Assign device based on rank
        device = torch.device(f"cuda:{dist.get_rank() % torch.cuda.device_count()}" if torch.cuda.is_available() else "cpu")
        logging.info(f"🖥️ Moving model components to {device}...")

        # Move components to the assigned device
        model.text_encoder.model = model.text_encoder.model.to(device).to(torch.float16)
        model.vae.model = model.vae.model.to(device).to(torch.float16)
        if hasattr(model, "clip") and hasattr(model.clip, "model"):
            model.clip.model = model.clip.model.to(device).to(torch.float32)
        else:
            logging.warning("⚠️ CLIP module not found. Skipping.")

        # Core model to FP16 and wrap with DDP if distributed
        model.model = model.model.to(device).to(torch.float16)
        if dist.is_initialized():
            model.model = torch.nn.parallel.DistributedDataParallel(
                model.model, device_ids=[device.index] if device.type == "cuda" else None, output_device=device.index if device.type == "cuda" else None
            )
            logging.info(f"🚀 Wrapped core model with DistributedDataParallel on {device}")
        else:
            logging.info(f"📌 Core model moved to {device} without DDP")

        MODEL_CACHE[key] = model
        logging.info(f"✅ {task} model loaded and distributed successfully.")
        log_gpu_usage()

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
    sample_steps: int = None
    sample_shift: float = 3.0
    sample_guide_scale: float = 3.0
    save_file: str = None
    image: str = None  # For I2V & T2I
    background_music: bool = True  # ✅ Enabled by default
    overlay_text: str = None  # ✅ Default text overlay - Can be "Generated by Wan2.1 AI" or whatever watermark text which sticks through all frames
    use_prompt_extend: bool = False  # ✅ Disabled by default
    prompt_extend_method: str = "local_qwen"
    prompt_extend_model: str = None
    prompt_extend_target_lang: str = "en"
    keyframe_interval: int = 30

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
            v = ((v // 4) * 4) + 1
            logging.warning(f"⚠️ Adjusted num_frames to {v} to satisfy 4n+1 constraint.")
        return v

    @model_validator(mode='after')
    def set_sample_steps_default(cls, values):
        """Set sample_steps based on task if not provided."""
        if values.sample_steps is None:  # ✅ Use dot notation
            if values.task.startswith('t2v'):
                values.sample_steps = 50
            elif values.task.startswith('i2v'):
                values.sample_steps = 40
        return values

def generate_video(request: VideoRequest):
    """Generate video based on the request parameters."""
    start_time = time.time()
    torch.manual_seed(request.seed)

    model = MODEL_CACHE[(request.task, request.t5_fsdp, request.dit_fsdp)]  # Retrieve the cached model

    # Handle T2I task right here
    if request.task == 't2i':
        with torch.no_grad():  # No gradients needed for inference
            image = model.generate(
                input_prompt=request.prompt,  # Corrected argument name
                size=request.size,  # Ensure this is a tuple (width, height)
                frame_num=1,  # T2I should only generate 1 frame
                shift=request.sample_shift or 5.0,
                sample_solver=request.sample_solver or 'unipc',
                sampling_steps=request.sample_steps or 40,
                guide_scale=request.sample_guide_scale or 5.0,
                n_prompt="",  # Set a default empty negative prompt
                seed=request.seed,
                offload_model=request.offload_model
            )[0]  # Get the first (and only) frame
        output_file = request.save_file or f"output_image_{int(time.time())}.png"
        image_np = image.cpu().numpy()
        # Normalize to 0-255 if needed, then save as PNG
        image_np = (image_np * 255).astype(np.uint8) if image_np.max() <= 1 else image_np
        cv2.imwrite(output_file, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
        return {"output_path": output_file, "time_seconds": time.time() - start_time}

    # Rest of your video generation logic for 't2v' and 'i2v' goes here
    # (e.g., keyframes, interpolation, audio, etc.)
    keyframe_interval = request.keyframe_interval
    num_keyframes = request.num_frames // keyframe_interval
    batch_size = get_optimal_batch_size()
    num_batches = (num_keyframes + batch_size - 1) // batch_size

    prompt = request.prompt
    keyframes = []

    for i in range(num_batches):
        batch_prompts = [prompt] * min(batch_size, num_keyframes - i * batch_size)

        with torch.no_grad():
            if request.task.startswith('i2v') and request.image:
                image_tensor = preprocess_image(request.image)
                batch_frames = model.generate(
                    input_prompt=batch_prompts,  # Corrected argument name
                    size=request.size,  # Ensure this is a tuple (width, height)
                    frame_num=request.num_frames,  # Corrected num_frames → frame_num
                    shift=request.sample_shift or 5.0,
                    sample_solver=request.sample_solver or 'unipc',
                    sampling_steps=request.sample_steps or 40,
                    guide_scale=request.sample_guide_scale or 5.0,
                    n_prompt="",  # Set a default empty negative prompt
                    seed=request.seed,
                    offload_model=request.offload_model,
                    image=image_tensor  # Only pass image for 'i2v' tasks
                )
            else:
                batch_frames = model.generate(
                    input_prompt=batch_prompts,  # Corrected argument name
                    size=request.size,  # Ensure this is a tuple (width, height)
                    frame_num=request.num_frames,  # Corrected num_frames → frame_num
                    shift=request.sample_shift or 5.0,
                    sample_solver=request.sample_solver or 'unipc',
                    sampling_steps=request.sample_steps or 40,
                    guide_scale=request.sample_guide_scale or 5.0,
                    n_prompt="",  # Set a default empty negative prompt
                    seed=request.seed,
                    offload_model=request.offload_model
                )

        keyframes.extend(batch_frames)

    all_frames = []
    for i in range(len(keyframes) - 1):
        all_frames.append(keyframes[i])
        all_frames.extend(interpolate_frames([keyframes[i], keyframes[i + 1]], keyframe_interval))
    all_frames.append(keyframes[-1])

    if request.background_music:
        logging.info("🎵 Generating Background Music...")
        audio_path = model.generate_audio(prompt=prompt)
    else:
        audio_path = None

    if request.overlay_text:
        logging.info("📝 Overlaying Text on Video...")
        all_frames = apply_text_overlay(all_frames, request.overlay_text)

    output_file = request.save_file or f"output_{int(time.time())}.mp4"
    Thread(target=save_video_async, args=(all_frames, output_file, request.fps, audio_path)).start()

    return {"output_path": output_file, "audio_path": audio_path, "time_seconds": time.time() - start_time}

def apply_text_overlay(frames, text, position="bottom", font_scale_factor=0.05):
    """Apply text overlay to each frame with improved scaling, positioning, and readability."""
    if not text:
        return frames  # Skip if no text is provided

    overlaid_frames = []
    for frame in frames:
        # Convert frame to NumPy array if it's a tensor
        if isinstance(frame, torch.Tensor):
            frame_np = frame.cpu().numpy()
            frame_np = (frame_np * 255).astype(np.uint8) if frame_np.max() <= 1 else frame_np
            frame_np = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        else:
            frame_np = frame

        # Get frame dimensions
        height, width, _ = frame_np.shape

        # Dynamically adjust font scale based on video resolution
        font_scale = max(1, int(font_scale_factor * height / 30))  # Auto-scale font size
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_color = (255, 255, 255)  # White text
        thickness = max(1, int(font_scale * 1.5))  # Scale thickness dynamically

        # Get text size to position dynamically
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_width, text_height = text_size

        # Dynamic text positioning
        if position == "top":
            x, y = 50, 50 + text_height
        elif position == "bottom":
            x, y = (width - text_width) // 2, height - 50
        else:  # Default: center
            x, y = (width - text_width) // 2, (height + text_height) // 2

        # Add semi-transparent background for better readability
        overlay = frame_np.copy()
        bg_x1, bg_y1 = x - 10, y - text_height - 10
        bg_x2, bg_y2 = x + text_width + 10, y + 10
        cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
        alpha = 0.5  # Transparency level
        frame_np = cv2.addWeighted(overlay, alpha, frame_np, 1 - alpha, 0)

        # Draw text on frame
        cv2.putText(frame_np, text, (x, y), font, font_scale, font_color, thickness, cv2.LINE_AA)

        # Convert back to RGB
        frame_rgb = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)
        overlaid_frames.append(frame_rgb)

    return overlaid_frames

def extend_prompt(prompt, method, model, target_lang):
    """Extend the input prompt using Local Qwen (default)."""
    return f"{prompt}"  # ✅ Keeps the original prompt without extra junk

def preprocess_image(image):
    """Convert image input into a PyTorch tensor for I2V tasks."""
    img = cv2.imread(image) if isinstance(image, str) else np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(img).permute(2, 0, 1).float().to(DEVICE) / 255.0

def raft_interpolation(keyframes, interval):
    """Interpolate frames using RAFT (GPU-accelerated optical flow)."""
    interpolated_frames = []

    for i in range(len(keyframes) - 1):
        frame1 = torch.tensor(keyframes[i]).permute(2, 0, 1).unsqueeze(0).float().cuda()
        frame2 = torch.tensor(keyframes[i + 1]).permute(2, 0, 1).unsqueeze(0).float().cuda()

        padder = InputPadder(frame1.shape)
        frame1, frame2 = padder.pad(frame1, frame2)

        with torch.no_grad():
            flow = raft_model(frame1, frame2, iters=20, test_mode=True)[0]  # Optical flow from frame1 → frame2

        H, W = frame1.shape[2:]  # Height & Width

        for j in range(1, interval):
            alpha = j / interval

            # Generate flow grid for warping
            grid_x, grid_y = torch.meshgrid(torch.arange(W), torch.arange(H))
            grid_x = grid_x.cuda().float()
            grid_y = grid_y.cuda().float()

            # Apply optical flow scaling based on alpha (progress)
            new_x = grid_x + alpha * flow[0, 0]  # X displacement
            new_y = grid_y + alpha * flow[0, 1]  # Y displacement

            # Normalize coordinates for grid_sample()
            new_x = (2.0 * new_x / (W - 1)) - 1.0
            new_y = (2.0 * new_y / (H - 1)) - 1.0
            grid = torch.stack((new_x, new_y), dim=-1).unsqueeze(0)

            # Warp frame1 towards frame2 using flow-based grid
            interp_frame = F.grid_sample(frame1, grid, mode="bilinear", padding_mode="border", align_corners=True)
            interp_frame = interp_frame.squeeze(0).permute(1, 2, 0).cpu().numpy()
            interpolated_frames.append(interp_frame)

    return interpolated_frames

def opencv_interpolation(keyframes, interval):
    """Interpolate frames using OpenCV's Farneback method for optical flow."""
    interpolated_frames = []
    for i in range(len(keyframes) - 1):
        frame1 = keyframes[i]
        frame2 = keyframes[i + 1]
        flow = cv2.calcOpticalFlowFarneback(cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY),
                                            cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY),
                                            None, 0.5, 3, 15, 3, 5, 1.2, 0)
        for j in range(1, interval):
            alpha = j / interval
            interp_frame = cv2.addWeighted(frame1, 1 - alpha, frame2, alpha, 0)
            interpolated_frames.append(interp_frame)
    return interpolated_frames

def interpolate_frames(keyframes, interval):
    """Choose RAFT or OpenCV for frame interpolation."""
    if USE_RAFT:
        logging.info("🚀 Using RAFT for GPU-accelerated frame interpolation.")
        return raft_interpolation(keyframes, interval)
    else:
        logging.info("⚠️ RAFT not available, using OpenCV optical flow instead.")
        return opencv_interpolation(keyframes, interval)

def save_video_async(frames, output_file, fps, audio_path=None):
    """Save video frames asynchronously with error handling."""
    try:
        writer = ffmpeg.write_frames(output_file, (1280, 720), fps=fps)
        writer.send(None)
        for frame in frames:
            writer.send(frame)
        writer.close()

        if audio_path:
            os.system(f"ffmpeg -i {output_file} -i {audio_path} -c:v copy -c:a aac {output_file.replace('.mp4', '_with_audio.mp4')}")

    except Exception as e:
        logging.error(f"❌ Error saving video: {e}")

def get_optimal_batch_size():
    """Dynamically adjust batch size based on available GPU memory."""
    total_memory = torch.cuda.get_device_properties(0).total_memory
    if total_memory > 40e9:  # A100 80GB
        return 128
    elif total_memory > 20e9:  # A40, RTX 3090
        return 64
    else:
        return 32  # Safe default


def log_gpu_usage():
    """Logs GPU memory usage and processes running on each GPU."""
    logging.info("📊 Checking GPU usage...")
    try:
        result = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, text=True)
        logging.info("\n" + result.stdout)  # Print entire nvidia-smi output
    except Exception as e:
        logging.error(f"❌ Failed to log GPU usage: {e}")

@app.on_event("startup")
async def startup_event():
    """Initialize distributed environment and load models during FastAPI startup."""
    logging.info("🚀 Starting up...")
    init_distributed()  # Initialize distributed process group

    # Preload models only on rank 0 to avoid redundant loading
    if not dist.is_initialized() or dist.get_rank() == 0:
        preload_tasks = [
            {"task": "t2v", "ckpt_dir": os.path.join(MODEL_ROOT, MODEL_WAN_T2V_14B), "huggingface_repo_id": "Wan-AI/Wan2.1-T2V-14B" ,"t5_fsdp": False, "dit_fsdp": False},
            # Add more as needed
        ]
        for config in preload_tasks:
            try:
                load_model(**config)
            except Exception as e:
                logging.error(f"❌ Failed to preload model for task={config['task']}: {e}")
    if dist.is_initialized():
        dist.barrier()  # Synchronize all processes after loading

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup the distributed environment."""
    logging.info("🚀 Shutting down and cleaning up...")
    if dist.is_initialized():
        dist.destroy_process_group()

@app.post("/generate/")
async def generate_api(request: VideoRequest):
    """Asynchronous endpoint for video generation."""
    try:
        result = await asyncio.to_thread(generate_video, request)
        return result
    except Exception as e:
        logging.error(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    if dist.is_initialized() and dist.get_rank() == 0:
        import uvicorn
        import os

        # Read options from environment variables with defaults
        workers = int(os.getenv("UVICORN_WORKERS", 1))
        reload = os.getenv("UVICORN_RELOAD", "false").lower() == "true"
        log_level = os.getenv("UVICORN_LOG_LEVEL", "info")

        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            workers=workers,
            reload=reload,
            log_level=log_level
        )
    elif dist.is_initialized():
        while True:
            time.sleep(1000)