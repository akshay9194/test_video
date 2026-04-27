"""
Marketing Video API — FastAPI server

Endpoints:
  POST /generate          — Generate a marketing video
  GET  /status/{job_id}   — Check job status
  GET  /health            — Health check
  GET  /voices            — List available TTS voices
"""

import asyncio
import os
import base64
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel
from loguru import logger

from marketing.config import PipelineConfig
from marketing.orchestrator import MarketingPipeline, VideoRequest, VideoResult


# Global state
pipeline: MarketingPipeline = None
active_jobs: dict[str, dict] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    logger.info("Starting Marketing Video Pipeline API...")
    config = PipelineConfig.from_env()
    pipeline = MarketingPipeline(config)
    pipeline.load_models()
    logger.info("API ready")
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="Marketing Video Generator",
    description="AI-powered marketing video creation from scripts and images",
    version="1.0.0",
    lifespan=lifespan,
)


# --- Request/Response Models ---

class GenerateRequest(BaseModel):
    script: str
    total_duration_sec: int = 30
    style: str = "photorealistic"
    aspect_ratio: str = "9:16"
    voice: str | None = None
    music_prompt: str = "upbeat corporate background music, inspiring, modern"
    add_captions: bool = True
    add_music: bool = True
    seed: int = 42


class GenerateResponse(BaseModel):
    job_id: str
    status: str
    message: str


class JobStatus(BaseModel):
    job_id: str
    status: str  # queued, processing, completed, failed
    result: dict | None = None
    error: str | None = None


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool


# --- Endpoints ---

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        models_loaded=pipeline is not None and pipeline.video_gen._loaded,
    )


@app.get("/voices")
async def list_voices():
    try:
        voices = await pipeline.tts.list_voices()
        return {"voices": voices}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate", response_model=GenerateResponse)
async def generate_video(request: GenerateRequest):
    """Submit a video generation job."""
    video_request = VideoRequest(
        script=request.script,
        total_duration_sec=request.total_duration_sec,
        style=request.style,
        aspect_ratio=request.aspect_ratio,
        voice=request.voice,
        music_prompt=request.music_prompt,
        add_captions=request.add_captions,
        add_music=request.add_music,
        seed=request.seed,
    )

    # Run in background
    job_id = None

    async def _run():
        try:
            active_jobs[job_id]["status"] = "processing"
            result = await pipeline.generate(video_request)
            active_jobs[job_id]["status"] = "completed"
            active_jobs[job_id]["result"] = {
                "output_path": result.output_path,
                "duration_sec": result.duration_sec,
                "scenes_count": result.scenes_count,
            }
        except Exception as e:
            logger.error(f"Job {job_id} failed: {e}")
            active_jobs[job_id]["status"] = "failed"
            active_jobs[job_id]["error"] = str(e)

    # Create job entry
    import uuid
    job_id = str(uuid.uuid4())[:8]
    active_jobs[job_id] = {"status": "queued", "result": None, "error": None}

    asyncio.create_task(_run())

    return GenerateResponse(
        job_id=job_id,
        status="queued",
        message="Video generation started. Poll /status/{job_id} for updates.",
    )


@app.post("/generate_sync")
async def generate_video_sync(request: GenerateRequest):
    """Synchronous video generation — waits and returns the result."""
    video_request = VideoRequest(
        script=request.script,
        total_duration_sec=request.total_duration_sec,
        style=request.style,
        aspect_ratio=request.aspect_ratio,
        voice=request.voice,
        music_prompt=request.music_prompt,
        add_captions=request.add_captions,
        add_music=request.add_music,
        seed=request.seed,
    )

    try:
        result = await pipeline.generate(video_request)

        # Read video and return as base64
        with open(result.output_path, "rb") as f:
            video_b64 = base64.b64encode(f.read()).decode()

        return {
            "job_id": result.job_id,
            "status": "completed",
            "video_base64": video_b64,
            "duration_sec": result.duration_sec,
            "scenes_count": result.scenes_count,
        }
    except Exception as e:
        logger.error(f"Sync generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status/{job_id}", response_model=JobStatus)
async def get_status(job_id: str):
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = active_jobs[job_id]
    return JobStatus(
        job_id=job_id,
        status=job["status"],
        result=job.get("result"),
        error=job.get("error"),
    )


@app.get("/download/{job_id}")
async def download_video(job_id: str):
    """Download the generated video file."""
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = active_jobs[job_id]
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Job status: {job['status']}")

    output_path = job["result"]["output_path"]
    if not os.path.exists(output_path):
        raise HTTPException(status_code=404, detail="Video file not found")

    return FileResponse(
        output_path,
        media_type="video/mp4",
        filename=f"marketing_{job_id}.mp4",
    )
