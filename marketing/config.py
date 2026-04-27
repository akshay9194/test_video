"""
Marketing Video Pipeline - Configuration

Controls which provider (local model vs API) is used for each component.
Set via environment variables or .env file.
"""

import os
from dataclasses import dataclass, field
from enum import Enum


class Provider(str, Enum):
    LOCAL = "local"
    API = "api"


@dataclass
class TTSConfig:
    provider: Provider = Provider.LOCAL
    # Local (edge-tts)
    local_voice: str = "en-US-JennyNeural"
    # Azure
    azure_speech_key: str = ""
    azure_speech_region: str = ""
    azure_voice: str = "en-US-JennyNeural"

    @classmethod
    def from_env(cls):
        return cls(
            provider=Provider(os.getenv("TTS_PROVIDER", "local")),
            local_voice=os.getenv("TTS_LOCAL_VOICE", "en-US-JennyNeural"),
            azure_speech_key=os.getenv("AZURE_SPEECH_KEY", ""),
            azure_speech_region=os.getenv("AZURE_SPEECH_REGION", ""),
            azure_voice=os.getenv("AZURE_TTS_VOICE", "en-US-JennyNeural"),
        )


@dataclass
class MusicConfig:
    provider: Provider = Provider.LOCAL
    # Local (MusicGen)
    model_size: str = "small"  # small, medium, large
    duration_sec: int = 30
    # API placeholder
    api_key: str = ""
    api_url: str = ""

    @classmethod
    def from_env(cls):
        return cls(
            provider=Provider(os.getenv("MUSIC_PROVIDER", "local")),
            model_size=os.getenv("MUSIC_MODEL_SIZE", "small"),
            duration_sec=int(os.getenv("MUSIC_DURATION_SEC", "30")),
            api_key=os.getenv("MUSIC_API_KEY", ""),
            api_url=os.getenv("MUSIC_API_URL", ""),
        )


@dataclass
class TalkingHeadConfig:
    provider: Provider = Provider.LOCAL
    # Local (SadTalker)
    sadtalker_checkpoint_dir: str = "/workspace/sadtalker/checkpoints"
    # API placeholder
    api_key: str = ""
    api_url: str = ""

    @classmethod
    def from_env(cls):
        return cls(
            provider=Provider(os.getenv("TALKING_HEAD_PROVIDER", "local")),
            sadtalker_checkpoint_dir=os.getenv("SADTALKER_CHECKPOINT_DIR", "/workspace/sadtalker/checkpoints"),
            api_key=os.getenv("TALKING_HEAD_API_KEY", ""),
            api_url=os.getenv("TALKING_HEAD_API_URL", ""),
        )


@dataclass
class VideoGenConfig:
    model_path: str = "/workspace/ckpts"
    resolution: str = "480p"
    default_video_length: int = 121
    default_aspect_ratio: str = "9:16"
    num_inference_steps: int = 30
    cfg_distilled: bool = True
    offloading: bool = True
    overlap_group_offloading: bool = True

    @classmethod
    def from_env(cls):
        return cls(
            model_path=os.getenv("MODEL_PATH", "/workspace/ckpts"),
            resolution=os.getenv("VIDEO_RESOLUTION", "480p"),
            default_video_length=int(os.getenv("VIDEO_LENGTH", "121")),
            default_aspect_ratio=os.getenv("VIDEO_ASPECT_RATIO", "9:16"),
            num_inference_steps=int(os.getenv("VIDEO_INFERENCE_STEPS", "30")),
            cfg_distilled=os.getenv("VIDEO_CFG_DISTILLED", "true").lower() == "true",
            offloading=os.getenv("VIDEO_OFFLOADING", "true").lower() == "true",
            overlap_group_offloading=os.getenv("VIDEO_OVERLAP_OFFLOADING", "true").lower() == "true",
        )


@dataclass
class CaptionConfig:
    enabled: bool = True
    whisper_model: str = "base"  # tiny, base, small, medium, large
    font_size: int = 24
    font_color: str = "white"
    bg_color: str = "black@0.6"
    position: str = "bottom"  # top, center, bottom

    @classmethod
    def from_env(cls):
        return cls(
            enabled=os.getenv("CAPTION_ENABLED", "true").lower() == "true",
            whisper_model=os.getenv("WHISPER_MODEL", "base"),
            font_size=int(os.getenv("CAPTION_FONT_SIZE", "24")),
            font_color=os.getenv("CAPTION_FONT_COLOR", "white"),
            position=os.getenv("CAPTION_POSITION", "bottom"),
        )


@dataclass
class SceneParserConfig:
    provider: Provider = Provider.LOCAL
    # API (Azure OpenAI / OpenAI)
    rewrite_provider: str = "azure"  # openai or azure
    openai_api_key: str = ""
    azure_openai_api_key: str = ""
    azure_openai_endpoint: str = ""
    azure_openai_deployment: str = ""
    azure_openai_api_version: str = "2024-06-01"
    model_name: str = "gpt-4o-mini"

    @classmethod
    def from_env(cls):
        return cls(
            provider=Provider(os.getenv("SCENE_PARSER_PROVIDER", "api")),
            rewrite_provider=os.getenv("REWRITE_PROVIDER", "azure"),
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            azure_openai_api_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
            azure_openai_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
            azure_openai_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", ""),
            azure_openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01"),
            model_name=os.getenv("SCENE_PARSER_MODEL", "gpt-4o-mini"),
        )


@dataclass
class PipelineConfig:
    tts: TTSConfig = field(default_factory=TTSConfig)
    music: MusicConfig = field(default_factory=MusicConfig)
    talking_head: TalkingHeadConfig = field(default_factory=TalkingHeadConfig)
    video_gen: VideoGenConfig = field(default_factory=VideoGenConfig)
    caption: CaptionConfig = field(default_factory=CaptionConfig)
    scene_parser: SceneParserConfig = field(default_factory=SceneParserConfig)

    # Global
    output_dir: str = "/workspace/outputs"
    temp_dir: str = "/workspace/temp"
    fps: int = 24

    @classmethod
    def from_env(cls):
        return cls(
            tts=TTSConfig.from_env(),
            music=MusicConfig.from_env(),
            talking_head=TalkingHeadConfig.from_env(),
            video_gen=VideoGenConfig.from_env(),
            caption=CaptionConfig.from_env(),
            scene_parser=SceneParserConfig.from_env(),
            output_dir=os.getenv("OUTPUT_DIR", "/workspace/outputs"),
            temp_dir=os.getenv("TEMP_DIR", "/workspace/temp"),
            fps=int(os.getenv("VIDEO_FPS", "24")),
        )
