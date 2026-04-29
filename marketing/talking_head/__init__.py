"""Talking head module"""
from marketing.talking_head.base import BaseTalkingHead
from marketing.talking_head.wav2lip import Wav2LipLocal
from marketing.talking_head.sadtalker import SadTalkerLocal
from marketing.talking_head.api_talking_head import TalkingHeadAPI


def create_talking_head(config) -> BaseTalkingHead:
    from marketing.config import Provider
    if config.provider == Provider.LOCAL:
        return Wav2LipLocal(config)
    elif config.provider == Provider.API:
        return TalkingHeadAPI(config)
    else:
        raise ValueError(f"Unknown talking head provider: {config.provider}")
