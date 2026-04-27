"""Talking head module"""
from marketing.talking_head.base import BaseTalkingHead
from marketing.talking_head.sadtalker import SadTalkerLocal
from marketing.talking_head.api_talking_head import TalkingHeadAPI


def create_talking_head(config) -> BaseTalkingHead:
    from marketing.config import Provider
    if config.provider == Provider.LOCAL:
        return SadTalkerLocal(config)
    elif config.provider == Provider.API:
        return TalkingHeadAPI(config)
    else:
        raise ValueError(f"Unknown talking head provider: {config.provider}")
