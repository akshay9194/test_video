"""TTS module"""
from marketing.tts.base import BaseTTS
from marketing.tts.local_tts import EdgeTTS
from marketing.tts.azure_tts import AzureTTS


def create_tts(config) -> BaseTTS:
    from marketing.config import Provider
    if config.provider == Provider.LOCAL:
        return EdgeTTS(config)
    elif config.provider == Provider.API:
        return AzureTTS(config)
    else:
        raise ValueError(f"Unknown TTS provider: {config.provider}")
