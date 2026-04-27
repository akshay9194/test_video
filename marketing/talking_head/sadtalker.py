"""SadTalker integration for talking head generation.

SadTalker: https://github.com/OpenTalker/SadTalker

Setup:
  git clone https://github.com/OpenTalker/SadTalker.git /workspace/sadtalker
  cd /workspace/sadtalker
  pip install -r requirements.txt
  bash scripts/download_models.sh
"""

import asyncio
import os
import subprocess
import sys
from loguru import logger
from marketing.talking_head.base import BaseTalkingHead


class SadTalkerLocal(BaseTalkingHead):
    def __init__(self, config):
        self.checkpoint_dir = config.sadtalker_checkpoint_dir
        self.sadtalker_dir = os.path.dirname(self.checkpoint_dir)

    def _check_installation(self):
        if not os.path.exists(self.sadtalker_dir):
            raise FileNotFoundError(
                f"SadTalker not found at {self.sadtalker_dir}. "
                f"Install it:\n"
                f"  git clone https://github.com/OpenTalker/SadTalker.git {self.sadtalker_dir}\n"
                f"  cd {self.sadtalker_dir}\n"
                f"  pip install -r requirements.txt\n"
                f"  bash scripts/download_models.sh"
            )

    async def generate(self, image_path: str, audio_path: str, output_path: str) -> str:
        self._check_installation()

        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)

        logger.info(f"SadTalker: Generating talking head from {image_path} + {audio_path}")

        cmd = [
            sys.executable,
            os.path.join(self.sadtalker_dir, "inference.py"),
            "--driven_audio", audio_path,
            "--source_image", image_path,
            "--result_dir", output_dir,
            "--checkpoint_dir", self.checkpoint_dir,
            "--size", "256",
            "--enhancer", "gfpgan",
            "--still",
        ]

        def _run():
            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd=self.sadtalker_dir
            )
            if result.returncode != 0:
                raise RuntimeError(f"SadTalker failed: {result.stderr}")

            # SadTalker outputs to result_dir with auto-generated name
            # Find the most recent .mp4 in output_dir
            mp4_files = sorted(
                [f for f in os.listdir(output_dir) if f.endswith(".mp4")],
                key=lambda f: os.path.getmtime(os.path.join(output_dir, f)),
                reverse=True,
            )
            if not mp4_files:
                raise FileNotFoundError("SadTalker produced no output video")

            generated = os.path.join(output_dir, mp4_files[0])
            if generated != output_path:
                os.rename(generated, output_path)

            logger.info(f"SadTalker: Saved to {output_path}")
            return output_path

        return await asyncio.to_thread(_run)
