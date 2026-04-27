"""
CLI tool to test the marketing pipeline without the API server.

Usage:
  python test_marketing.py --script "Welcome to Le Grand Bleu Hotel..." --duration 15
"""

import asyncio
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from marketing.config import PipelineConfig
from marketing.orchestrator import MarketingPipeline, VideoRequest


async def main():
    parser = argparse.ArgumentParser(description="Generate marketing video from script")
    parser.add_argument("--script", type=str, required=True, help="Marketing script text")
    parser.add_argument("--duration", type=int, default=15, help="Total video duration in seconds")
    parser.add_argument("--style", type=str, default="photorealistic", help="Visual style")
    parser.add_argument("--aspect_ratio", type=str, default="9:16", help="Aspect ratio")
    parser.add_argument("--voice", type=str, default=None, help="TTS voice name")
    parser.add_argument("--music_prompt", type=str, default="upbeat corporate background music, inspiring",
                        help="Music generation prompt")
    parser.add_argument("--no_captions", action="store_true", help="Disable captions")
    parser.add_argument("--no_music", action="store_true", help="Disable background music")
    parser.add_argument("--avatar", type=str, default=None, help="Path to avatar image for talking head")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default=None, help="Output path")
    args = parser.parse_args()

    config = PipelineConfig.from_env()
    if args.output:
        config.output_dir = os.path.dirname(args.output) or config.output_dir

    pipeline = MarketingPipeline(config)
    pipeline.load_models()

    request = VideoRequest(
        script=args.script,
        total_duration_sec=args.duration,
        style=args.style,
        aspect_ratio=args.aspect_ratio,
        avatar_image=args.avatar,
        voice=args.voice,
        music_prompt=args.music_prompt,
        add_captions=not args.no_captions,
        add_music=not args.no_music,
        seed=args.seed,
    )

    result = await pipeline.generate(request)
    print(f"\n{'='*60}")
    print(f"Marketing video generated!")
    print(f"  Output: {result.output_path}")
    print(f"  Duration: {result.duration_sec:.1f}s")
    print(f"  Scenes: {result.scenes_count}")
    print(f"  Job ID: {result.job_id}")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
