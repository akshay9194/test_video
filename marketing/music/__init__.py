"""Music generation module"""
from marketing.music.base import BaseMusic
from marketing.music.local_music import MusicGenLocal
from marketing.music.api_music import MusicAPI


def create_music_gen(config) -> BaseMusic:
    from marketing.config import Provider
    if config.provider == Provider.LOCAL:
        return MusicGenLocal(config)
    elif config.provider == Provider.API:
        return MusicAPI(config)
    else:
        raise ValueError(f"Unknown music provider: {config.provider}")
