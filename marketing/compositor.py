"""
Video Compositor — merges scenes, audio, music, and captions into a final video.

Uses FFmpeg for all video/audio operations.
"""

import asyncio
import os
import subprocess
from loguru import logger


class VideoCompositor:
    def __init__(self, fps: int = 24):
        self.fps = fps
        self._check_ffmpeg()

    def _check_ffmpeg(self):
        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        except FileNotFoundError:
            raise RuntimeError("ffmpeg not found. Install it: apt install ffmpeg")

    async def concat_videos(self, video_paths: list[str], output_path: str) -> str:
        """Concatenate multiple video clips into one."""
        if len(video_paths) == 1:
            # Just copy
            await self._run_ffmpeg([
                "ffmpeg", "-y", "-i", video_paths[0],
                "-c", "copy", output_path
            ])
            return output_path

        # Create concat file
        concat_file = output_path + ".concat.txt"
        with open(concat_file, "w") as f:
            for vp in video_paths:
                f.write(f"file '{os.path.abspath(vp)}'\n")

        # Need to re-encode for concat to work with different source files
        await self._run_ffmpeg([
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0", "-i", concat_file,
            "-c:v", "libx264", "-preset", "fast",
            "-pix_fmt", "yuv420p",
            "-r", str(self.fps),
            output_path,
        ])

        os.unlink(concat_file)
        logger.info(f"Concatenated {len(video_paths)} clips -> {output_path}")
        return output_path

    async def add_audio(self, video_path: str, audio_path: str,
                        output_path: str, volume: float = 1.0) -> str:
        """Add audio track to a video."""
        await self._run_ffmpeg([
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", audio_path,
            "-filter_complex", f"[1:a]volume={volume}[a]",
            "-map", "0:v", "-map", "[a]",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            output_path,
        ])
        logger.info(f"Added audio to video -> {output_path}")
        return output_path

    async def mix_audio_tracks(self, video_path: str, voiceover_path: str,
                                music_path: str, output_path: str,
                                voice_volume: float = 1.0,
                                music_volume: float = 0.15) -> str:
        """Mix voiceover and background music, add to video."""
        await self._run_ffmpeg([
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", voiceover_path,
            "-i", music_path,
            "-filter_complex",
            f"[1:a]volume={voice_volume}[voice];"
            f"[2:a]volume={music_volume}[music];"
            f"[voice][music]amix=inputs=2:duration=first:dropout_transition=3[a]",
            "-map", "0:v", "-map", "[a]",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            output_path,
        ])
        logger.info(f"Mixed audio tracks -> {output_path}")
        return output_path

    async def add_subtitles(self, video_path: str, srt_path: str,
                            output_path: str, font_size: int = 24,
                            font_color: str = "white",
                            position: str = "bottom") -> str:
        """Burn subtitles into video."""
        if position == "bottom":
            margin_v = 30
        elif position == "top":
            margin_v = 30
        else:
            margin_v = 0

        alignment = {"bottom": 2, "top": 6, "center": 5}.get(position, 2)

        subtitle_filter = (
            f"subtitles={srt_path}:force_style="
            f"'FontSize={font_size},"
            f"PrimaryColour=&H00FFFFFF,"
            f"OutlineColour=&H00000000,"
            f"BorderStyle=3,"
            f"Outline=1,"
            f"Shadow=0,"
            f"MarginV={margin_v},"
            f"Alignment={alignment}'"
        )

        await self._run_ffmpeg([
            "ffmpeg", "-y",
            "-i", video_path,
            "-vf", subtitle_filter,
            "-c:v", "libx264", "-preset", "fast",
            "-c:a", "copy",
            output_path,
        ])
        logger.info(f"Added subtitles -> {output_path}")
        return output_path

    async def trim_video(self, video_path: str, output_path: str,
                         duration_sec: float) -> str:
        """Trim video to exact duration."""
        await self._run_ffmpeg([
            "ffmpeg", "-y",
            "-i", video_path,
            "-t", str(duration_sec),
            "-c", "copy",
            output_path,
        ])
        return output_path

    async def resize_video(self, video_path: str, output_path: str,
                           width: int, height: int) -> str:
        """Resize video to specific dimensions."""
        await self._run_ffmpeg([
            "ffmpeg", "-y",
            "-i", video_path,
            "-vf", f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
                   f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2",
            "-c:v", "libx264", "-preset", "fast",
            "-c:a", "copy",
            output_path,
        ])
        return output_path

    async def get_duration(self, file_path: str) -> float:
        """Get duration of a video or audio file."""
        result = await self._run_ffmpeg([
            "ffprobe", "-v", "quiet",
            "-show_entries", "format=duration",
            "-of", "csv=p=0",
            file_path,
        ], capture=True)
        return float(result.strip())

    async def _run_ffmpeg(self, cmd: list[str], capture: bool = False):
        def _run():
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"FFmpeg failed: {' '.join(cmd)}\n{result.stderr}")
            return result.stdout

        return await asyncio.to_thread(_run)
