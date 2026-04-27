"""Azure Cognitive Services TTS"""
import asyncio
from loguru import logger
from marketing.tts.base import BaseTTS


class AzureTTS(BaseTTS):
    def __init__(self, config):
        self.speech_key = config.azure_speech_key
        self.speech_region = config.azure_speech_region
        self.default_voice = config.azure_voice

        if not self.speech_key or not self.speech_region:
            raise EnvironmentError(
                "AZURE_SPEECH_KEY and AZURE_SPEECH_REGION must be set for Azure TTS."
            )

    async def synthesize(self, text: str, output_path: str, voice: str = None) -> str:
        import azure.cognitiveservices.speech as speechsdk

        voice = voice or self.default_voice
        logger.info(f"AzureTTS: Generating speech with voice={voice}, length={len(text)} chars")

        speech_config = speechsdk.SpeechConfig(
            subscription=self.speech_key,
            region=self.speech_region,
        )
        speech_config.speech_synthesis_voice_name = voice
        audio_config = speechsdk.audio.AudioOutputConfig(filename=output_path)

        synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=speech_config,
            audio_config=audio_config,
        )

        result = await asyncio.to_thread(
            lambda: synthesizer.speak_text_async(text).get()
        )

        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            logger.info(f"AzureTTS: Saved audio to {output_path}")
            return output_path
        else:
            raise RuntimeError(f"Azure TTS failed: {result.reason}")

    async def list_voices(self) -> list[dict]:
        import azure.cognitiveservices.speech as speechsdk

        speech_config = speechsdk.SpeechConfig(
            subscription=self.speech_key,
            region=self.speech_region,
        )
        synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)
        result = await asyncio.to_thread(
            lambda: synthesizer.get_voices_async().get()
        )
        return [
            {
                "id": v.short_name,
                "name": v.local_name,
                "gender": str(v.gender),
                "locale": v.locale,
            }
            for v in result.voices
        ]
