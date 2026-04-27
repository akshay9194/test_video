"""Base TTS interface"""
from abc import ABC, abstractmethod


class BaseTTS(ABC):
    @abstractmethod
    async def synthesize(self, text: str, output_path: str, voice: str = None) -> str:
        """Convert text to speech audio file. Returns path to output file."""
        pass

    @abstractmethod
    async def list_voices(self) -> list[dict]:
        """List available voices."""
        pass
