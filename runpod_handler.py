"""
RunPod Serverless handler for HunyuanVideo-1.5 video generation.
Receives requests via RunPod's serverless API and returns generated video as base64.
"""

import os
import io
import base64
import datetime
import copy
import tempfile

# Set CUDA memory config before any torch imports
if 'PYTORCH_CUDA_ALLOC_CONF' not in os.environ:
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
import einops
import imageio
import runpod
from loguru import logger

from hyvideo.pipelines.hunyuan_video_pipeline import HunyuanVideo_1_5_Pipeline
from hyvideo.commons.parallel_states import initialize_parallel_state
from hyvideo.commons.infer_state import InferState

# Initialize parallel state (single GPU)
parallel_dims = initialize_parallel_state(sp=1)
torch.cuda.set_device(0)

# ---- Global pipeline (loaded once at cold start) ----
PIPE = None
ENABLE_OFFLOADING = True
ENABLE_GROUP_OFFLOADING = True
OVERLAP_GROUP_OFFLOADING = True

MODEL_PATH = os.getenv("MODEL_PATH", "/workspace/ckpts")


def load_pipeline():
    """Load the pipeline once during container startup."""
    global PIPE, ENABLE_OFFLOADING, ENABLE_GROUP_OFFLOADING, OVERLAP_GROUP_OFFLOADING

    logger.info("Loading HunyuanVideo-1.5 pipeline...")

    transformer_version = HunyuanVideo_1_5_Pipeline.get_transformer_version(
        "480p", "t2v", False, False, False
    )
    transformer_dtype = torch.bfloat16

    # Offloading config
    ENABLE_OFFLOADING = True
    offloading_config = HunyuanVideo_1_5_Pipeline.get_offloading_config()
    ENABLE_GROUP_OFFLOADING = offloading_config.get('enable_group_offloading', True)
    OVERLAP_GROUP_OFFLOADING = True

    device = torch.device('cpu')  # offloading: load to CPU first
    transformer_init_device = torch.device('cpu')

    PIPE = HunyuanVideo_1_5_Pipeline.create_pipeline(
        pretrained_model_name_or_path=MODEL_PATH,
        transformer_version=transformer_version,
        create_sr_pipeline=False,  # SR disabled to save memory
        transformer_dtype=transformer_dtype,
        device=device,
        transformer_init_device=transformer_init_device,
    )

    infer_state = InferState(
        enable_cache=True,
        cache_type="deepcache",
        no_cache_block_id=list(range(53, 54)),
        cache_start_step=11,
        cache_end_step=45,
        total_steps=50,
        cache_step_interval=4,
    )

    PIPE.apply_infer_optimization(
        infer_state=infer_state,
        enable_offloading=ENABLE_OFFLOADING,
        enable_group_offloading=ENABLE_GROUP_OFFLOADING,
        overlap_group_offloading=OVERLAP_GROUP_OFFLOADING,
    )

    logger.info("Pipeline loaded successfully.")


def video_to_base64(video_tensor, fps=24):
    """Convert a video tensor to base64-encoded mp4."""
    if video_tensor.ndim == 5:
        video_tensor = video_tensor[0]

    vid = (video_tensor * 255).clamp(0, 255).to(torch.uint8)
    vid = einops.rearrange(vid, 'c f h w -> f h w c')
    frames = [frame.cpu().numpy() for frame in vid]

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        imageio.mimwrite(tmp_path, frames, fps=fps)
        with open(tmp_path, "rb") as f:
            video_bytes = f.read()
        return base64.b64encode(video_bytes).decode("utf-8")
    finally:
        os.unlink(tmp_path)


def handler(job):
    """
    RunPod serverless handler.

    Expected input:
    {
        "prompt": "A sleek smartphone rotating on a white table...",
        "negative_prompt": "",          # optional
        "aspect_ratio": "16:9",         # optional, default "16:9"
        "video_length": 121,            # optional, 121-721, default 121
        "num_inference_steps": 50,      # optional, default 50
        "seed": 123,                    # optional
        "rewrite": true,               # optional, default true
        "image_path_base64": null       # optional, base64-encoded ref image for i2v
    }
    """
    job_input = job["input"]

    prompt = job_input.get("prompt")
    if not prompt:
        return {"error": "prompt is required"}

    negative_prompt = job_input.get("negative_prompt", "")
    aspect_ratio = job_input.get("aspect_ratio", "16:9")
    video_length = job_input.get("video_length", 121)
    num_inference_steps = job_input.get("num_inference_steps", 50)
    seed = job_input.get("seed", 123)
    enable_rewrite = job_input.get("rewrite", True)

    # Validate video_length
    if video_length < 121 or video_length > 721:
        return {"error": f"video_length must be between 121 (5 sec) and 721 (30 sec), got {video_length}"}
    if (video_length - 1) % 4 != 0:
        return {"error": f"video_length must satisfy (N-1) % 4 == 0. Valid: 121, 241, 361, 481, 601, 721"}

    extra_kwargs = {}

    # Handle I2V: decode base64 image to temp file
    ref_image_path = None
    image_base64 = job_input.get("image_path_base64")
    if image_base64:
        try:
            img_bytes = base64.b64decode(image_base64)
            tmp_img = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            tmp_img.write(img_bytes)
            tmp_img.close()
            ref_image_path = tmp_img.name
            extra_kwargs["reference_image"] = ref_image_path
        except Exception as e:
            return {"error": f"Failed to decode reference image: {e}"}

    try:
        logger.info(f"Generating video: prompt='{prompt[:80]}...', length={video_length}, steps={num_inference_steps}")

        out = PIPE(
            enable_sr=False,
            prompt=prompt,
            aspect_ratio=aspect_ratio,
            num_inference_steps=num_inference_steps,
            sr_num_inference_steps=None,
            video_length=video_length,
            negative_prompt=negative_prompt,
            seed=seed,
            output_type="pt",
            prompt_rewrite=enable_rewrite,
            return_pre_sr_video=False,
            **extra_kwargs,
        )

        video_b64 = video_to_base64(out.videos)

        duration_sec = round(video_length / 24, 1)
        return {
            "video_base64": video_b64,
            "video_length_frames": video_length,
            "video_duration_sec": duration_sec,
            "resolution": "480p",
            "format": "mp4",
        }

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        return {"error": str(e)}

    finally:
        # Cleanup temp ref image
        if ref_image_path and os.path.exists(ref_image_path):
            os.unlink(ref_image_path)


# Load pipeline at container startup
load_pipeline()

# Start RunPod serverless worker
runpod.serverless.start({"handler": handler})
