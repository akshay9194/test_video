"""Base music generation interface"""
from abc import ABC, abstractmethod


class BaseMusic(ABC):
    @abstractmethod
    async def generate(self, prompt: str, duration_sec: int, output_path: str) -> str:
        """Generate background music. Returns path to output wav/mp3."""
        pass
