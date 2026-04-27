"""
HunyuanVideo wrapper for the marketing pipeline.

Wraps the existing generate.py pipeline into a callable Python API.
"""

import os
import sys
import threading
import torch
import copy
from loguru import logger

# Ensure parent package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Initialize parallel state ONCE at module level (must happen before any threads)
_init_done = False
_init_lock = threading.Lock()

def _ensure_init():
    global _init_done
    if _init_done:
        return
    with _init_lock:
        if _init_done:
            return
        from hyvideo.commons.parallel_states import initialize_parallel_state
        initialize_parallel_state(sp=int(os.environ.get('WORLD_SIZE', '1')))
        torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', '0')))
        _init_done = True


class HunyuanVideoGenerator:
    """Wraps HunyuanVideo-1.5 pipeline for programmatic use."""

    def __init__(self, config):
        self.config = config
        self.pipe = None
        self._loaded = False
        self._load_lock = threading.Lock()

    def load(self):
        """Load the pipeline (call once at startup). Thread-safe."""
        if self._loaded:
            return

        with self._load_lock:
            if self._loaded:
                return

            _ensure_init()

            from hyvideo.pipelines.hunyuan_video_pipeline import HunyuanVideo_1_5_Pipeline
            from hyvideo.commons.infer_state import InferState

            task = "t2v"
            transformer_version = HunyuanVideo_1_5_Pipeline.get_transformer_version(
                self.config.resolution, task, self.config.cfg_distilled, False, False
            )

            transformer_dtype = torch.bfloat16
            device = torch.device("cpu") if self.config.offloading else torch.device("cuda")
            transformer_init_device = torch.device("cpu")

            logger.info(f"Loading HunyuanVideo pipeline from {self.config.model_path}")

            self.pipe = HunyuanVideo_1_5_Pipeline.create_pipeline(
                pretrained_model_name_or_path=self.config.model_path,
                transformer_version=transformer_version,
                create_sr_pipeline=False,
                transformer_dtype=transformer_dtype,
                device=device,
                transformer_init_device=transformer_init_device,
            )

            offloading_config = HunyuanVideo_1_5_Pipeline.get_offloading_config()
            enable_group_offloading = offloading_config.get("enable_group_offloading", True)

            infer_state = InferState(
                enable_cache=False,
            )

            self.pipe.apply_infer_optimization(
                infer_state=infer_state,
                enable_offloading=self.config.offloading,
                enable_group_offloading=enable_group_offloading,
                overlap_group_offloading=self.config.overlap_group_offloading,
            )

            self._loaded = True
            logger.info("HunyuanVideo pipeline loaded successfully")

    def generate(
        self,
        prompt: str,
        output_path: str,
        video_length: int = None,
        aspect_ratio: str = None,
        seed: int = 42,
        num_inference_steps: int = None,
        reference_image: str = None,
    ) -> str:
        """Generate a video clip. Returns path to the output mp4."""
        if not self._loaded:
            self.load()

        import einops
        import imageio

        video_length = video_length or self.config.default_video_length
        aspect_ratio = aspect_ratio or self.config.default_aspect_ratio
        num_inference_steps = num_inference_steps or self.config.num_inference_steps

        # Validate video_length
        if (video_length - 1) % 4 != 0:
            # Round to nearest valid value
            video_length = ((video_length - 1) // 4) * 4 + 1
            logger.warning(f"Rounded video_length to {video_length}")

        video_length = max(121, min(721, video_length))

        extra_kwargs = {}
        if reference_image:
            extra_kwargs["reference_image"] = reference_image

        logger.info(f"Generating video: prompt='{prompt[:80]}...', length={video_length}, steps={num_inference_steps}")

        out = self.pipe(
            enable_sr=False,
            prompt=prompt,
            aspect_ratio=aspect_ratio,
            num_inference_steps=num_inference_steps,
            sr_num_inference_steps=None,
            video_length=video_length,
            negative_prompt="",
            seed=seed,
            output_type="pt",
            prompt_rewrite=False,
            return_pre_sr_video=False,
            **extra_kwargs,
        )

        # Save video
        video = out.videos
        if video.ndim == 5:
            video = video[0]
        vid = (video * 255).clamp(0, 255).to(torch.uint8)
        vid = einops.rearrange(vid, "c f h w -> f h w c")

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        imageio.mimwrite(output_path, vid.cpu().numpy(), fps=24)

        logger.info(f"Video saved to {output_path}")
        return output_path

    def generate_for_duration(
        self,
        prompt: str,
        output_path: str,
        duration_sec: int = 5,
        aspect_ratio: str = None,
        seed: int = 42,
        reference_image: str = None,
    ) -> str:
        """Generate video for a specified duration in seconds."""
        # Convert seconds to frames (24 fps), must satisfy (N-1) % 4 == 0
        raw_frames = duration_sec * 24
        video_length = ((raw_frames - 1) // 4) * 4 + 1
        video_length = max(121, min(721, video_length))

        return self.generate(
            prompt=prompt,
            output_path=output_path,
            video_length=video_length,
            aspect_ratio=aspect_ratio,
            seed=seed,
            reference_image=reference_image,
        )
