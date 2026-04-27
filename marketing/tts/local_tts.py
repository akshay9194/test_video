"""Local TTS using edge-tts (free, no API key needed)"""
import asyncio
from loguru import logger
from marketing.tts.base import BaseTTS


class EdgeTTS(BaseTTS):
    def __init__(self, config):
        self.default_voice = config.local_voice

    async def synthesize(self, text: str, output_path: str, voice: str = None) -> str:
        import edge_tts

        voice = voice or self.default_voice
        logger.info(f"EdgeTTS: Generating speech with voice={voice}, length={len(text)} chars")

        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(output_path)

        logger.info(f"EdgeTTS: Saved audio to {output_path}")
        return output_path

    async def list_voices(self) -> list[dict]:
        import edge_tts

        voices = await edge_tts.list_voices()
        return [
            {
                "id": v["ShortName"],
                "name": v["FriendlyName"],
                "gender": v["Gender"],
                "locale": v["Locale"],
            }
            for v in voices
        ]
