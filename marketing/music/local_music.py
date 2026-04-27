"""Local music generation using Meta's MusicGen (audiocraft)"""
import asyncio
import torch
from loguru import logger
from marketing.music.base import BaseMusic


class MusicGenLocal(BaseMusic):
    def __init__(self, config):
        self.model_size = config.model_size
        self.model = None

    def _load_model(self):
        if self.model is not None:
            return

        from audiocraft.models import MusicGen as MusicGenModel

        logger.info(f"Loading MusicGen model ({self.model_size})...")
        self.model = MusicGenModel.get_pretrained(f"facebook/musicgen-{self.model_size}")
        logger.info("MusicGen loaded")

    async def generate(self, prompt: str, duration_sec: int, output_path: str) -> str:
        def _generate():
            self._load_model()

            self.model.set_generation_params(duration=duration_sec)
            logger.info(f"MusicGen: Generating {duration_sec}s music for prompt: '{prompt[:60]}...'")

            wav = self.model.generate([prompt])

            # wav shape: (batch, channels, samples)
            import torchaudio
            torchaudio.save(output_path, wav[0].cpu(), sample_rate=32000)
            logger.info(f"MusicGen: Saved to {output_path}")
            return output_path

        return await asyncio.to_thread(_generate)
