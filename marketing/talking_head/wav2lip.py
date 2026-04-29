"""Wav2Lip integration for talking head generation.

Wav2Lip: https://github.com/Rudrabha/Wav2Lip

Setup:
  cd /workspace
  git clone https://github.com/Rudrabha/Wav2Lip.git
  cd Wav2Lip
  mkdir -p checkpoints
  # Download wav2lip_gan.pth from:
  # https://github.com/Rudrabha/Wav2Lip#getting-the-weights
  # Place it at /workspace/Wav2Lip/checkpoints/wav2lip_gan.pth
"""

import asyncio
import os
import subprocess
import sys
from loguru import logger
from marketing.talking_head.base import BaseTalkingHead


class Wav2LipLocal(BaseTalkingHead):
    def __init__(self, config):
        self.wav2lip_dir = os.getenv("WAV2LIP_DIR", "/workspace/Wav2Lip")
        self.checkpoint_path = os.path.join(self.wav2lip_dir, "checkpoints", "wav2lip_gan.pth")
        # Use Wav2Lip's own venv if available
        self.python_path = os.path.join(self.wav2lip_dir, "venv", "bin", "python")
        if not os.path.exists(self.python_path):
            self.python_path = sys.executable

    def _check_installation(self):
        if not os.path.exists(self.wav2lip_dir):
            raise FileNotFoundError(
                f"Wav2Lip not found at {self.wav2lip_dir}. "
                f"Install it:\n"
                f"  cd /workspace\n"
                f"  git clone https://github.com/Rudrabha/Wav2Lip.git\n"
                f"  cd Wav2Lip\n"
                f"  python3 -m venv venv\n"
                f"  source venv/bin/activate\n"
                f"  pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124\n"
                f"  pip install numpy==1.26.4 opencv-python-headless==4.10.0.84 librosa==0.9.2 scipy\n"
                f"  pip install face-detection\n"
                f"  mkdir -p checkpoints\n"
                f"  # Download wav2lip_gan.pth into checkpoints/"
            )
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(
                f"Wav2Lip checkpoint not found at {self.checkpoint_path}. "
                f"Download wav2lip_gan.pth from https://github.com/Rudrabha/Wav2Lip#getting-the-weights "
                f"and place it at {self.checkpoint_path}"
            )

    async def generate(self, image_path: str, audio_path: str, output_path: str) -> str:
        self._check_installation()

        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)

        logger.info(f"Wav2Lip: Generating talking head from {image_path} + {audio_path}")

        # Wav2Lip inference command
        cmd = [
            self.python_path,
            os.path.join(self.wav2lip_dir, "inference.py"),
            "--checkpoint_path", self.checkpoint_path,
            "--face", image_path,
            "--audio", audio_path,
            "--outfile", output_path,
            "--nosmooth",
            "--pads", "0", "10", "0", "0",
        ]

        def _run():
            env = os.environ.copy()
            env["PYTHONPATH"] = self.wav2lip_dir

            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd=self.wav2lip_dir, env=env
            )
            if result.returncode != 0:
                raise RuntimeError(f"Wav2Lip failed: {result.stderr[-500:]}")

            if not os.path.exists(output_path):
                # Wav2Lip might save to results/result_voice.mp4
                default_output = os.path.join(self.wav2lip_dir, "results", "result_voice.mp4")
                if os.path.exists(default_output):
                    import shutil
                    shutil.move(default_output, output_path)
                else:
                    raise FileNotFoundError("Wav2Lip produced no output video")

            logger.info(f"Wav2Lip: Saved to {output_path}")
            return output_path

        return await asyncio.to_thread(_run)
