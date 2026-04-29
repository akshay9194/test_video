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
- "broll": A visual scene generated from text description. Use when NO reference images are available for this part.
- "image_video": Animate a provided reference image. Use this whenever a reference image matches the scene content. Provide description of desired motion/animation.
- "talking_head": A person speaks to camera. ONLY use this if explicitly requested AND an avatar image is confirmed available.

IMPORTANT RULES FOR IMAGE USAGE:
- When reference images are provided, PREFER "image_video" scenes over "broll" scenes.
- Match each reference image to the most relevant part of the script.
- Each image should be used as a scene if possible.
- Set "image_ref" to the exact filename of the image to use.
- For image_video, the "video_prompt" should describe MOTION only (the image provides the visuals): camera movement, character actions, animations.
- Do NOT use "talking_head" unless explicitly told an avatar is available.

Output a JSON array of scenes. Each scene has:
- "type": "broll" | "talking_head" | "image_video"
- "duration_sec": estimated duration in seconds (5-15)
- "script_text": the narration text for this scene (voiceover plays over ALL scenes)
- "video_prompt": detailed visual description for video generation (for broll) OR motion description (for image_video)
- "image_ref": filename if an image is referenced, else null

Rules:
- Total duration should match the requested video length
- Distribute the script text across scenes as narration (every scene gets a portion of the script as voiceover)
- For broll prompts, follow this formula: Subject + Motion + Scene + Camera Movement + Lighting + Style
- For image_video prompts, describe motion: "The person gestures with their hand", "The camera slowly zooms in", "Charts animate and numbers increase"
- Use concrete visual descriptions, not abstract words
- Camera movements: "The camera slowly moves forward", "The camera pans left", "The camera orbits around"
- Always end the video prompt with "Photorealistic style." unless a different style is requested

Example output with images:
[
  {"type": "image_video", "duration_sec": 5, "script_text": "Struggling with low conversions?", "video_prompt": "The man shakes his head slightly, his fingers type on the laptop. The red graph on screen pulses. The camera slowly pushes in toward the screen. Cinematic lighting.", "image_ref": "frustrated_man.jpg"},
  {"type": "image_video", "duration_sec": 5, "script_text": "Meet BizAgent.ai, your autonomous sales executive.", "video_prompt": "The dashboard metrics animate, bar charts rise, numbers increase. The camera slowly pans right across the curved screen. Blue neon glow.", "image_ref": "dashboard.jpg"},
  {"type": "image_video", "duration_sec": 5, "script_text": "Sarah handles calls and converts leads around the clock.", "video_prompt": "The holographic woman gestures with her right hand, presenting the floating charts. Subtle holographic flicker. The camera slowly orbits around her.", "image_ref": "sarah_hologram.jpg"},
  {"type": "image_video", "duration_sec": 5, "script_text": "BizAgent.ai. Your business, on autopilot.", "video_prompt": "The logo subtly pulses with a cyan glow. The camera slowly zooms in. Dark background with soft light rays.", "image_ref": "logo.jpg"}
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

        # Extract just filenames for the LLM (not full paths)
        if images:
            image_filenames = [os.path.basename(img) for img in images]
            image_list = ", ".join(image_filenames)
            image_instruction = (
                f"Available reference images: {image_list}\n"
                f"IMPORTANT: Use 'image_video' type for scenes that match these images. "
                f"Set 'image_ref' to the exact filename. Prefer image_video over broll when an image matches."
            )
        else:
            image_instruction = "No reference images provided. Use only 'broll' type scenes."

        user_prompt = (
            f"Script: {script}\n"
            f"Total video duration: {total_duration_sec} seconds\n"
            f"Style: {style}\n"
            f"{image_instruction}\n"
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
