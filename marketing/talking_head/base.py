"""Base talking head interface"""
from abc import ABC, abstractmethod


class BaseTalkingHead(ABC):
    @abstractmethod
    async def generate(self, image_path: str, audio_path: str, output_path: str) -> str:
        """Generate a talking head video from a face image and audio.
        
        Args:
            image_path: Path to the face/avatar image
            audio_path: Path to the speech audio file
            output_path: Where to save the output video

        Returns:
            Path to the generated video
        """
        pass
