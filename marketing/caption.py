"""
Caption generation using OpenAI Whisper for timing + FFmpeg for rendering.
"""

import asyncio
import json
import os
import subprocess
from loguru import logger


class CaptionGenerator:
    def __init__(self, config):
        self.config = config
        self.whisper_model = None

    def _load_whisper(self):
        if self.whisper_model is not None:
            return

        import whisper
        logger.info(f"Loading Whisper model ({self.config.whisper_model})...")
        self.whisper_model = whisper.load_model(self.config.whisper_model)
        logger.info("Whisper loaded")

    async def generate_subtitles(self, audio_path: str, output_srt_path: str) -> str:
        """Transcribe audio and generate SRT subtitle file."""
        def _transcribe():
            self._load_whisper()
            logger.info(f"Transcribing {audio_path}...")
            result = self.whisper_model.transcribe(audio_path, word_timestamps=True)
            return result

        result = await asyncio.to_thread(_transcribe)

        # Convert to SRT format
        srt_content = self._to_srt(result)
        with open(output_srt_path, "w", encoding="utf-8") as f:
            f.write(srt_content)

        logger.info(f"Subtitles saved to {output_srt_path}")
        return output_srt_path

    async def generate_from_script(self, script: str, audio_duration_sec: float,
                                    output_srt_path: str) -> str:
        """Generate simple timed subtitles from script text without Whisper.
        Splits script into chunks and spaces them evenly across the duration."""
        words = script.split()
        chunk_size = max(5, len(words) // max(1, int(audio_duration_sec / 3)))
        chunks = []

        for i in range(0, len(words), chunk_size):
            chunks.append(" ".join(words[i:i + chunk_size]))

        time_per_chunk = audio_duration_sec / max(len(chunks), 1)
        srt_lines = []

        for i, chunk in enumerate(chunks):
            start = i * time_per_chunk
            end = min((i + 1) * time_per_chunk, audio_duration_sec)
            srt_lines.append(f"{i + 1}")
            srt_lines.append(f"{self._format_time(start)} --> {self._format_time(end)}")
            srt_lines.append(chunk)
            srt_lines.append("")

        with open(output_srt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(srt_lines))

        logger.info(f"Script-based subtitles saved to {output_srt_path}")
        return output_srt_path

    def _to_srt(self, whisper_result) -> str:
        segments = whisper_result.get("segments", [])
        srt_lines = []

        for i, seg in enumerate(segments):
            start = seg["start"]
            end = seg["end"]
            text = seg["text"].strip()
            srt_lines.append(f"{i + 1}")
            srt_lines.append(f"{self._format_time(start)} --> {self._format_time(end)}")
            srt_lines.append(text)
            srt_lines.append("")

        return "\n".join(srt_lines)

    @staticmethod
    def _format_time(seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
