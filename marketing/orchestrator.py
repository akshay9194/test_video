"""
Marketing Video Pipeline — Orchestrator

Ties all components together:
  Script → Scene Parser → [TTS + Video Gen + Talking Head + Music] → Compositor → Final Video
"""

import asyncio
import os
import shutil
import time
import uuid
from dataclasses import dataclass, field
from loguru import logger

from marketing.config import PipelineConfig
from marketing.scene_parser import SceneParser, Scene
from marketing.video_gen import HunyuanVideoGenerator
from marketing.compositor import VideoCompositor
from marketing.caption import CaptionGenerator
from marketing.tts import create_tts
from marketing.music import create_music_gen
from marketing.talking_head import create_talking_head


@dataclass
class VideoRequest:
    script: str
    total_duration_sec: int = 30
    style: str = "photorealistic"
    aspect_ratio: str = "9:16"
    avatar_image: str | None = None  # Path to avatar face image for talking_head scenes
    images: list[str] | None = None  # Reference images for image_video scenes
    voice: str | None = None
    music_prompt: str = "upbeat corporate background music, inspiring, modern"
    add_captions: bool = True
    add_music: bool = True
    seed: int = 42


@dataclass
class VideoResult:
    output_path: str
    duration_sec: float
    scenes_count: int
    job_id: str
    step_timings: dict = field(default_factory=dict)


# Shared state for live progress from the UI
_progress_callbacks: dict[str, callable] = {}


def set_progress_callback(job_id: str, callback: callable):
    _progress_callbacks[job_id] = callback


def _report_progress(job_id: str, step: str, status: str, elapsed: float = 0):
    logger.info(f"[{job_id}] {step}: {status} ({elapsed:.1f}s)")
    cb = _progress_callbacks.get(job_id)
    if cb:
        cb(step, status, elapsed)


class MarketingPipeline:
    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig.from_env()
        os.makedirs(self.config.output_dir, exist_ok=True)
        os.makedirs(self.config.temp_dir, exist_ok=True)

        # Initialize components
        self.scene_parser = SceneParser(self.config.scene_parser)
        self.tts = create_tts(self.config.tts)
        self.music_gen = create_music_gen(self.config.music)
        self.talking_head = create_talking_head(self.config.talking_head)
        self.caption = CaptionGenerator(self.config.caption)
        self.compositor = VideoCompositor(fps=self.config.fps)
        self.video_gen = HunyuanVideoGenerator(self.config.video_gen)

    def load_models(self):
        """Pre-load T2V model (call at startup). I2V loads on demand."""
        logger.info("Loading video generation model (T2V)...")
        self.video_gen.load(task="t2v")
        logger.info("All models ready")

    async def generate(self, request: VideoRequest) -> VideoResult:
        """Main pipeline: script → final video with audio."""
        job_id = str(uuid.uuid4())[:8]
        job_dir = os.path.join(self.config.temp_dir, job_id)
        os.makedirs(job_dir, exist_ok=True)
        step_timings = {}
        pipeline_start = time.time()

        logger.info(f"[{job_id}] Starting marketing video generation")
        logger.info(f"[{job_id}] Script: '{request.script[:100]}...'")
        logger.info(f"[{job_id}] Duration: {request.total_duration_sec}s, Style: {request.style}")

        try:
            # Step 1: Parse script into scenes
            t0 = time.time()
            _report_progress(job_id, "Scene Parsing", "running")
            scenes = await self.scene_parser.parse(
                script=request.script,
                total_duration_sec=request.total_duration_sec,
                style=request.style,
                images=request.images,
            )
            step_timings["1_scene_parsing"] = round(time.time() - t0, 1)
            _report_progress(job_id, "Scene Parsing", f"done ({len(scenes)} scenes)", step_timings["1_scene_parsing"])

            # Step 2: Generate full voiceover
            t0 = time.time()
            _report_progress(job_id, "Voiceover (TTS)", "running")
            full_script = " ".join(s.script_text for s in scenes if s.script_text)
            voiceover_path = os.path.join(job_dir, "voiceover.wav")

            if full_script.strip():
                await self.tts.synthesize(full_script, voiceover_path, voice=request.voice)
                voiceover_duration = await self.compositor.get_duration(voiceover_path)
            else:
                voiceover_path = None
                voiceover_duration = request.total_duration_sec
            step_timings["2_tts"] = round(time.time() - t0, 1)
            _report_progress(job_id, "Voiceover (TTS)", "done", step_timings["2_tts"])

            # Step 3: Generate video clips for each scene
            t0 = time.time()
            clip_paths = []

            for i, scene in enumerate(scenes):
                scene_start = time.time()
                _report_progress(job_id, f"Video Scene {i+1}/{len(scenes)}", f"running ({scene.type})")
                clip_path = os.path.join(job_dir, f"scene_{i:02d}.mp4")

                if scene.type == "broll":
                    self.video_gen.generate_for_duration(
                        prompt=scene.video_prompt,
                        output_path=clip_path,
                        duration_sec=scene.duration_sec,
                        aspect_ratio=request.aspect_ratio,
                        seed=request.seed + i,
                    )

                elif scene.type == "talking_head":
                    talking_head_success = False
                    if request.avatar_image:
                        try:
                            scene_audio = os.path.join(job_dir, f"scene_{i:02d}_audio.wav")
                            await self.tts.synthesize(scene.script_text, scene_audio, voice=request.voice)
                            await self.talking_head.generate(
                                image_path=request.avatar_image,
                                audio_path=scene_audio,
                                output_path=clip_path,
                            )
                            talking_head_success = True
                        except Exception as e:
                            logger.warning(f"[{job_id}] Talking head failed: {e}. Trying I2V fallback.")

                    if not talking_head_success and request.avatar_image:
                        # Try I2V: animate the avatar image with motion
                        try:
                            motion_prompt = (
                                f"A person in a corporate setting speaks to the camera and gestures with their hands. "
                                f"{scene.script_text} Warm lighting, shallow depth of field. {request.style} style."
                            )
                            self.video_gen.generate_for_duration(
                                prompt=motion_prompt,
                                output_path=clip_path,
                                duration_sec=scene.duration_sec,
                                aspect_ratio=request.aspect_ratio,
                                seed=request.seed + i,
                                reference_image=request.avatar_image,
                            )
                            talking_head_success = True
                        except Exception as e:
                            logger.warning(f"[{job_id}] I2V avatar fallback also failed: {e}")

                    if not talking_head_success:
                        fallback_prompt = (
                            f"{scene.script_text} "
                            f"The camera slowly moves forward. Warm natural lighting. "
                            f"{request.style} style."
                        )
                        logger.info(f"[{job_id}] Using B-roll fallback for talking_head scene")
                        self.video_gen.generate_for_duration(
                            prompt=fallback_prompt,
                            output_path=clip_path,
                            duration_sec=scene.duration_sec,
                            aspect_ratio=request.aspect_ratio,
                            seed=request.seed + i,
                        )

                elif scene.type == "image_video":
                    ref_image = None
                    if scene.image_ref and request.images:
                        for img in request.images:
                            if os.path.basename(img) == scene.image_ref:
                                ref_image = img
                                break
                    if not ref_image and request.images:
                        ref_image = request.images[0]

                    if ref_image:
                        try:
                            self.video_gen.generate_for_duration(
                                prompt=scene.video_prompt or "The camera slowly pans across the scene.",
                                output_path=clip_path,
                                duration_sec=scene.duration_sec,
                                aspect_ratio=request.aspect_ratio,
                                seed=request.seed + i,
                                reference_image=ref_image,
                            )
                        except Exception as e:
                            logger.warning(f"[{job_id}] I2V failed: {e}. Falling back to T2V.")
                            self.video_gen.generate_for_duration(
                                prompt=f"{scene.video_prompt or ''} {request.style} style.",
                                output_path=clip_path,
                                duration_sec=scene.duration_sec,
                                aspect_ratio=request.aspect_ratio,
                                seed=request.seed + i,
                            )
                    else:
                        self.video_gen.generate_for_duration(
                            prompt=f"{scene.video_prompt or 'The camera slowly pans across the scene.'} {request.style} style.",
                            output_path=clip_path,
                            duration_sec=scene.duration_sec,
                            aspect_ratio=request.aspect_ratio,
                            seed=request.seed + i,
                        )

                clip_paths.append(clip_path)
                scene_elapsed = round(time.time() - scene_start, 1)
                step_timings[f"3_scene_{i+1}_{scene.type}"] = scene_elapsed
                _report_progress(job_id, f"Video Scene {i+1}/{len(scenes)}", "done", scene_elapsed)

            step_timings["3_video_gen_total"] = round(time.time() - t0, 1)

            # Step 4: Concatenate all clips
            t0 = time.time()
            _report_progress(job_id, "Concatenation", "running")
            concat_path = os.path.join(job_dir, "concat.mp4")
            await self.compositor.concat_videos(clip_paths, concat_path)
            step_timings["4_concat"] = round(time.time() - t0, 1)
            _report_progress(job_id, "Concatenation", "done", step_timings["4_concat"])

            # Step 5: Add audio
            t0 = time.time()
            current_video = concat_path

            if request.add_music:
                _report_progress(job_id, "Music Generation", "running")
                music_path = os.path.join(job_dir, "music.wav")
                video_duration = await self.compositor.get_duration(current_video)

                try:
                    await self.music_gen.generate(
                        prompt=request.music_prompt,
                        duration_sec=int(video_duration) + 2,
                        output_path=music_path,
                    )
                except Exception as e:
                    logger.warning(f"[{job_id}] Music generation failed: {e}. Continuing without music.")
                    music_path = None
            else:
                music_path = None

            _report_progress(job_id, "Audio Mixing", "running")
            if voiceover_path and music_path:
                with_audio = os.path.join(job_dir, "with_audio.mp4")
                await self.compositor.mix_audio_tracks(
                    current_video, voiceover_path, music_path, with_audio
                )
                current_video = with_audio
            elif voiceover_path:
                with_audio = os.path.join(job_dir, "with_audio.mp4")
                await self.compositor.add_audio(current_video, voiceover_path, with_audio)
                current_video = with_audio
            elif music_path:
                with_audio = os.path.join(job_dir, "with_audio.mp4")
                await self.compositor.add_audio(
                    current_video, music_path, with_audio, volume=0.3
                )
                current_video = with_audio
            step_timings["5_audio"] = round(time.time() - t0, 1)
            _report_progress(job_id, "Audio Mixing", "done", step_timings["5_audio"])

            # Step 6: Add captions
            t0 = time.time()
            if request.add_captions and full_script.strip():
                _report_progress(job_id, "Captions", "running")
                srt_path = os.path.join(job_dir, "captions.srt")

                if voiceover_path:
                    await self.caption.generate_subtitles(voiceover_path, srt_path)
                else:
                    total_dur = await self.compositor.get_duration(current_video)
                    await self.caption.generate_from_script(full_script, total_dur, srt_path)

                with_captions = os.path.join(job_dir, "with_captions.mp4")
                await self.compositor.add_subtitles(
                    current_video, srt_path, with_captions,
                    font_size=self.config.caption.font_size,
                    position=self.config.caption.position,
                )
                current_video = with_captions
                step_timings["6_captions"] = round(time.time() - t0, 1)
                _report_progress(job_id, "Captions", "done", step_timings["6_captions"])

            # Step 7: Copy to final output
            _report_progress(job_id, "Finalizing", "running")
            final_output = os.path.join(
                self.config.output_dir, f"marketing_{job_id}.mp4"
            )
            shutil.copy2(current_video, final_output)

            final_duration = await self.compositor.get_duration(final_output)
            step_timings["total"] = round(time.time() - pipeline_start, 1)
            _report_progress(job_id, "Finalizing", f"done ({final_duration:.1f}s video)", step_timings["total"])

            # Cleanup progress callback
            _progress_callbacks.pop(job_id, None)

            return VideoResult(
                output_path=final_output,
                duration_sec=final_duration,
                scenes_count=len(scenes),
                job_id=job_id,
                step_timings=step_timings,
            )

        finally:
            # Cleanup temp files
            try:
                shutil.rmtree(job_dir)
            except Exception:
                pass
