"""
Scene Parser - breaks a marketing script into individual scenes.

Each scene is either:
- "broll": a text-to-video scene (HunyuanVideo generates visuals)
- "talking_head": an avatar speaks a portion of the script
- "image_video": animate a provided image (I2V)
"""

import json
import os
from dataclasses import dataclass
from loguru import logger
import openai


SCENE_PARSER_SYSTEM_PROMPT = """You are a video production assistant. Given a marketing script and optional scene instructions, break the script into a sequence of video scenes.

Each scene must be one of these types:
- "broll": A visual scene with no speaking. Provide a detailed video generation prompt.
- "talking_head": A person speaks to camera. Provide the dialogue text.
- "image_video": Animate a provided image. Provide description of motion.

Output a JSON array of scenes. Each scene has:
- "type": "broll" | "talking_head" | "image_video"
- "duration_sec": estimated duration in seconds (5-15)
- "script_text": the spoken text (for talking_head) or empty string
- "video_prompt": detailed visual description for video generation (for broll/image_video)
- "image_ref": filename if an image is referenced, else null

Rules:
- Total duration should match the requested video length
- For broll prompts, follow this formula: Subject + Motion + Scene + Camera Movement + Lighting + Style
- Use concrete visual descriptions, not abstract words
- Camera movements: "The camera slowly moves forward", "The camera pans left", "The camera orbits around"
- Always end the video prompt with "Photorealistic style." unless a different style is requested
- Keep talking_head segments short (5-10 seconds each)

Example output:
[
  {"type": "broll", "duration_sec": 5, "script_text": "", "video_prompt": "A wide aerial shot descending toward a beachfront resort at golden hour. The camera tilts down revealing an infinity pool. Warm golden sunlight. Photorealistic style.", "image_ref": null},
  {"type": "talking_head", "duration_sec": 5, "script_text": "Welcome to Le Grand Bleu Hotel, where luxury meets the sea.", "video_prompt": "", "image_ref": null},
  {"type": "broll", "duration_sec": 5, "script_text": "", "video_prompt": "A smooth dolly shot through a hotel suite. White marble floors, king-size bed with rose petals. The camera moves forward toward balcony doors. Warm side lighting. Photorealistic style.", "image_ref": null}
]

Return ONLY the JSON array, no other text."""


@dataclass
class Scene:
    type: str  # broll, talking_head, image_video
    duration_sec: int
    script_text: str
    video_prompt: str
    image_ref: str | None = None


class SceneParser:
    def __init__(self, config):
        self.config = config

    async def parse(self, script: str, total_duration_sec: int = 30,
                    style: str = "photorealistic", images: list[str] = None) -> list[Scene]:
        from marketing.config import Provider

        if self.config.provider == Provider.API:
            return await self._parse_with_llm(script, total_duration_sec, style, images)
        else:
            return self._parse_simple(script, total_duration_sec, style, images)

    async def _parse_with_llm(self, script: str, total_duration_sec: int,
                               style: str, images: list[str] = None) -> list[Scene]:
        """Use OpenAI/Azure to intelligently break script into scenes."""
        if self.config.rewrite_provider == "azure":
            client = openai.AzureOpenAI(
                api_key=self.config.azure_openai_api_key,
                azure_endpoint=self.config.azure_openai_endpoint,
                api_version=self.config.azure_openai_api_version,
                timeout=120,
            )
            model = self.config.azure_openai_deployment
        else:
            client = openai.OpenAI(
                api_key=self.config.openai_api_key,
                timeout=120,
            )
            model = self.config.model_name

        image_list = ", ".join(images) if images else "none provided"
        user_prompt = (
            f"Script: {script}\n"
            f"Total video duration: {total_duration_sec} seconds\n"
            f"Style: {style}\n"
            f"Available images: {image_list}\n"
            f"Break this into scenes."
        )

        logger.info(f"SceneParser: Calling LLM to parse script ({len(script)} chars)")

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SCENE_PARSER_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=2000,
        )

        content = response.choices[0].message.content.strip()
        # Handle markdown code blocks
        if content.startswith("```"):
            content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        scenes_data = json.loads(content)
        scenes = [Scene(**s) for s in scenes_data]

        logger.info(f"SceneParser: Generated {len(scenes)} scenes")
        return scenes

    def _parse_simple(self, script: str, total_duration_sec: int,
                      style: str, images: list[str] = None) -> list[Scene]:
        """Simple rule-based parser when no LLM is available."""
        sentences = [s.strip() for s in script.replace(".", ".\n").split("\n") if s.strip()]
        scenes = []
        time_per_scene = max(5, total_duration_sec // max(len(sentences), 1))

        for i, sentence in enumerate(sentences):
            if i % 2 == 0:
                # Alternate between talking_head and broll
                scenes.append(Scene(
                    type="talking_head",
                    duration_sec=time_per_scene,
                    script_text=sentence,
                    video_prompt="",
                ))
            else:
                scenes.append(Scene(
                    type="broll",
                    duration_sec=time_per_scene,
                    script_text="",
                    video_prompt=f"{sentence} The camera slowly moves forward. Warm lighting. {style} style.",
                ))

        logger.info(f"SceneParser (simple): Generated {len(scenes)} scenes")
        return scenes
