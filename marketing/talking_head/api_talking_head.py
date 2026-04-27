"""API-based talking head generation (placeholder for HeyGen, D-ID, etc.)"""
from loguru import logger
from marketing.talking_head.base import BaseTalkingHead


class TalkingHeadAPI(BaseTalkingHead):
    def __init__(self, config):
        self.api_key = config.api_key
        self.api_url = config.api_url

    async def generate(self, image_path: str, audio_path: str, output_path: str) -> str:
        raise NotImplementedError(
            "Talking head API integration not yet implemented. "
            "Set TALKING_HEAD_PROVIDER=local to use SadTalker, "
            "or integrate HeyGen/D-ID API here."
        )
