"""API-based music generation (placeholder for Suno, etc.)"""
import asyncio
from loguru import logger
from marketing.music.base import BaseMusic


class MusicAPI(BaseMusic):
    def __init__(self, config):
        self.api_key = config.api_key
        self.api_url = config.api_url

    async def generate(self, prompt: str, duration_sec: int, output_path: str) -> str:
        # Placeholder for API integration (Suno, Udio, etc.)
        raise NotImplementedError(
            "Music API integration not yet implemented. "
            "Set MUSIC_PROVIDER=local to use MusicGen, or integrate your preferred music API here."
        )
