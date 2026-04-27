"""
Gradio UI for the Marketing Video Pipeline.

Features:
- Script input, style/duration/aspect controls
- Live progress log showing each step + timing
- System monitor: CPU, RAM, VRAM utilization
- Video preview after generation

Usage:
  pip install gradio
  python ui.py
"""

import asyncio
import os
import sys
import threading
import time

import gradio as gr
import psutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from marketing.config import PipelineConfig
from marketing.orchestrator import MarketingPipeline, VideoRequest, set_progress_callback


# --- System Monitor ---

def get_system_stats() -> dict:
    cpu_percent = psutil.cpu_percent(interval=0.1)
    mem = psutil.virtual_memory()
    ram_used_gb = round(mem.used / (1024 ** 3), 1)
    ram_total_gb = round(mem.total / (1024 ** 3), 1)
    ram_percent = mem.percent

    gpu_stats = {"gpu_name": "N/A", "vram_used_gb": 0, "vram_total_gb": 0, "vram_percent": 0, "gpu_util": 0}
    try:
        import torch
        if torch.cuda.is_available():
            gpu_stats["gpu_name"] = torch.cuda.get_device_name(0)
            vram_total = torch.cuda.get_device_properties(0).total_mem
            vram_used = torch.cuda.memory_allocated(0)
            vram_reserved = torch.cuda.memory_reserved(0)
            gpu_stats["vram_used_gb"] = round(vram_reserved / (1024 ** 3), 1)
            gpu_stats["vram_total_gb"] = round(vram_total / (1024 ** 3), 1)
            gpu_stats["vram_percent"] = round(vram_reserved / vram_total * 100, 1) if vram_total > 0 else 0

            # Try nvidia-smi for GPU utilization
            try:
                import subprocess
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                     "--format=csv,nounits,noheader"],
                    capture_output=True, text=True
                )
                if result.returncode == 0:
                    parts = result.stdout.strip().split(",")
                    gpu_stats["gpu_util"] = int(parts[0].strip())
                    gpu_stats["vram_used_gb"] = round(int(parts[1].strip()) / 1024, 1)
                    gpu_stats["vram_total_gb"] = round(int(parts[2].strip()) / 1024, 1)
                    gpu_stats["vram_percent"] = round(gpu_stats["vram_used_gb"] / gpu_stats["vram_total_gb"] * 100, 1)
            except Exception:
                pass
    except Exception:
        pass

    return {
        "cpu_percent": cpu_percent,
        "ram_used_gb": ram_used_gb,
        "ram_total_gb": ram_total_gb,
        "ram_percent": ram_percent,
        **gpu_stats,
    }


def format_system_stats(stats: dict) -> str:
    bar = lambda pct: "█" * int(pct / 5) + "░" * (20 - int(pct / 5))

    return f"""```
┌─────────────────────────────────────────────┐
│  SYSTEM MONITOR                             │
├─────────────────────────────────────────────┤
│  CPU:  [{bar(stats['cpu_percent'])}] {stats['cpu_percent']:5.1f}%  │
│  RAM:  [{bar(stats['ram_percent'])}] {stats['ram_used_gb']}/{stats['ram_total_gb']} GB │
│  GPU:  {stats['gpu_name']:<38s}│
│  VRAM: [{bar(stats['vram_percent'])}] {stats['vram_used_gb']}/{stats['vram_total_gb']} GB │
│  GPU%: [{bar(stats['gpu_util'])}] {stats['gpu_util']:5.1f}%  │
└─────────────────────────────────────────────┘
```"""


def format_timings_table(timings: dict) -> str:
    if not timings:
        return ""

    lines = ["```", "┌──────────────────────────────┬──────────┐", "│  Step                        │  Time    │",
             "├──────────────────────────────┼──────────┤"]

    for key, val in timings.items():
        label = key.split("_", 1)[1] if "_" in key else key
        label = label.replace("_", " ").title()
        lines.append(f"│  {label:<28s}│  {val:>5.1f}s  │")

    lines.append("└──────────────────────────────┴──────────┘")
    lines.append("```")
    return "\n".join(lines)


# --- Pipeline ---

pipeline: MarketingPipeline = None
pipeline_loading = False
_pipeline_lock = threading.Lock()


def ensure_pipeline():
    global pipeline, pipeline_loading
    with _pipeline_lock:
        if pipeline is not None:
            return
        if pipeline_loading:
            return
        pipeline_loading = True

    try:
        config = PipelineConfig.from_env()
        p = MarketingPipeline(config)
        p.load_models()
        with _pipeline_lock:
            pipeline = p
            pipeline_loading = False
    except Exception as e:
        with _pipeline_lock:
            pipeline_loading = False
        raise


# --- Gradio Handlers ---

def generate_video(script, duration, style, aspect_ratio, avatar_image, ref_images,
                   voice, music_prompt, add_captions, add_music, seed,
                   progress=gr.Progress(track_tqdm=True)):
    ensure_pipeline()

    progress_log = []
    job_id_holder = [None]

    def on_progress(step, status, elapsed):
        timestamp = time.strftime("%H:%M:%S")
        icon = "✅" if "done" in status else "⏳" if "running" in status else "❌"
        line = f"[{timestamp}] {icon} {step}: {status}"
        if elapsed > 0 and "done" in status:
            line += f" ({elapsed:.1f}s)"
        progress_log.append(line)

    # Collect reference image paths
    image_paths = None
    if ref_images:
        image_paths = [f.name if hasattr(f, 'name') else f for f in ref_images]

    request = VideoRequest(
        script=script,
        total_duration_sec=int(duration),
        style=style,
        aspect_ratio=aspect_ratio,
        avatar_image=avatar_image,
        images=image_paths,
        voice=voice if voice else None,
        music_prompt=music_prompt,
        add_captions=add_captions,
        add_music=add_music,
        seed=int(seed),
    )

    # Run async pipeline in sync context
    loop = asyncio.new_event_loop()

    def run():
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(pipeline.generate(request))

    # Set up progress callback after we know the job_id
    # We patch it by setting a global callback for any job
    from marketing import orchestrator
    original_report = orchestrator._report_progress

    def patched_report(job_id, step, status, elapsed=0):
        on_progress(step, status, elapsed)
        original_report(job_id, step, status, elapsed)

    orchestrator._report_progress = patched_report

    try:
        result = run()
    finally:
        orchestrator._report_progress = original_report
        loop.close()

    log_text = "\n".join(progress_log)
    timings_text = format_timings_table(result.step_timings)
    stats = format_system_stats(get_system_stats())

    summary = f"""### Generation Complete!
- **Output:** `{result.output_path}`
- **Duration:** {result.duration_sec:.1f}s
- **Scenes:** {result.scenes_count}
- **Total time:** {result.step_timings.get('total', 0):.1f}s
"""

    return result.output_path, log_text, timings_text, stats, summary


def refresh_monitor():
    return format_system_stats(get_system_stats())


# --- UI Layout ---

with gr.Blocks(
    title="Marketing Video Generator",
) as demo:
    gr.Markdown("# 🎬 Marketing Video Generator")
    gr.Markdown("Generate complete marketing videos from scripts — with voiceover, captions, and background music.")

    with gr.Row():
        # Left column: inputs
        with gr.Column(scale=2):
            script_input = gr.Textbox(
                label="Marketing Script",
                placeholder="Welcome to Le Grand Bleu Hotel. Experience luxury where the Mediterranean meets the sky...",
                lines=6,
            )

            with gr.Row():
                duration_input = gr.Slider(5, 30, value=15, step=5, label="Duration (seconds)")
                seed_input = gr.Number(value=42, label="Seed", precision=0)

            with gr.Row():
                style_input = gr.Dropdown(
                    choices=["photorealistic", "cinematic", "anime", "watercolor", "cyberpunk"],
                    value="photorealistic",
                    label="Visual Style",
                )
                aspect_input = gr.Dropdown(
                    choices=["9:16", "16:9", "1:1", "4:3"],
                    value="9:16",
                    label="Aspect Ratio",
                )

            with gr.Accordion("Images & Avatar", open=False):
                avatar_input = gr.Image(
                    label="Avatar Image (for talking head scenes)",
                    type="filepath",
                )
                images_input = gr.File(
                    label="Reference Images (for image-to-video scenes)",
                    file_count="multiple",
                    file_types=["image"],
                )

            voice_input = gr.Textbox(
                label="TTS Voice (optional)",
                placeholder="en-US-JennyNeural (leave blank for default)",
                value="",
            )
            music_prompt = gr.Textbox(
                label="Music Style",
                value="upbeat corporate background music, inspiring, modern",
            )

            with gr.Row():
                captions_check = gr.Checkbox(value=True, label="Add Captions")
                music_check = gr.Checkbox(value=False, label="Add Music (requires audiocraft)")

            generate_btn = gr.Button("🎬 Generate Video", variant="primary", size="lg")

        # Right column: outputs
        with gr.Column(scale=3):
            video_output = gr.Video(label="Generated Video")
            summary_output = gr.Markdown(label="Summary")

            with gr.Accordion("Step Timings", open=True):
                timings_output = gr.Textbox(label="", lines=12, interactive=False, elem_classes=["timing-box"])

            with gr.Accordion("Progress Log", open=False):
                log_output = gr.Textbox(label="", lines=15, interactive=False, elem_classes=["log-box"])

            with gr.Accordion("System Monitor", open=True):
                monitor_output = gr.Markdown()
                refresh_btn = gr.Button("🔄 Refresh", size="sm")
                refresh_btn.click(fn=refresh_monitor, outputs=[monitor_output])

    # Wire up
    generate_btn.click(
        fn=generate_video,
        inputs=[script_input, duration_input, style_input, aspect_input,
                avatar_input, images_input,
                voice_input, music_prompt, captions_check, music_check, seed_input],
        outputs=[video_output, log_output, timings_output, monitor_output, summary_output],
    )

    # Auto-refresh monitor on load
    demo.load(fn=refresh_monitor, outputs=[monitor_output])


if __name__ == "__main__":
    # Pipeline loads on first generate request (not in background — avoids thread race)
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        allowed_paths=["/workspace/outputs", "/workspace/temp", "/tmp"],
    )
