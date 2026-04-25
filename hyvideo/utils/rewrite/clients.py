# -*- coding: utf-8 -*-
# Licensed under the TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/main/LICENSE
#
# Unless and only to the extent required by applicable law, the Tencent Hunyuan works and any
# output and results therefrom are provided "AS IS" without any express or implied warranties of
# any kind including any warranties of title, merchantability, noninfringement, course of dealing,
# usage of trade, or fitness for a particular purpose. You are solely responsible for determining the
# appropriateness of using, reproducing, modifying, performing, displaying or distributing any of
# the Tencent Hunyuan works or outputs and assume any and all risks associated with your or a
# third party's use or distribution of any of the Tencent Hunyuan works or outputs and your exercise
# of rights and permissions under this agreement.
# See the License for the specific language governing permissions and limitations under the License.

import time
import os
import io
import base64

import openai
from loguru import logger


def _get_rewrite_provider():
    """Get the configured rewrite provider from environment variables."""
    return os.getenv("REWRITE_PROVIDER", "openai").lower()


class OpenAIClient(object):
    """Client for OpenAI API (text-only prompt rewriting)."""

    def __init__(self, model_name=None):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise EnvironmentError("OPENAI_API_KEY must be set in environment variables.")
        self.model_name = model_name or os.getenv("REWRITE_MODEL_NAME", "gpt-4o")

    def _api_call(self, system_prompt: str, user_input: str, temperature: float, max_tokens: int):
        client = openai.OpenAI(api_key=self.api_key, timeout=600)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ]

        last_err = None
        for i in range(5):
            try:
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return response.choices[0].message.content or ""
            except Exception as e:
                last_err = e
                if i < 4:
                    time.sleep(2 ** i)
                    continue
                raise last_err

    def run_single_recaption(self, system_prompt, input_prompt, temperature=0.1, max_tokens=4096):
        return self._api_call(system_prompt, input_prompt, temperature, max_tokens)


class OpenAIVisionClient(object):
    """Client for OpenAI API with vision support (image-to-video prompt rewriting)."""

    def __init__(self, model_name=None):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise EnvironmentError("OPENAI_API_KEY must be set in environment variables.")
        self.model_name = model_name or os.getenv("REWRITE_MODEL_NAME", "gpt-4o")
        self.max_image_size = int(os.getenv("I2V_REWRITE_MAX_IMAGE_SIZE", "1024"))

    def _encode_image_to_base64(self, image_path: str, max_dimension: int) -> str:
        try:
            from PIL import Image
        except ImportError as e:
            logger.error("Pillow (PIL) is required for image encoding but is not installed.")
            raise e

        try:
            image = image_path
            if not isinstance(image, Image.Image):
                image = Image.open(image)

            with image as img:
                if img.width > max_dimension or img.height > max_dimension:
                    img.thumbnail((max_dimension, max_dimension))

                buffer = io.BytesIO()
                if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
                    img = img.convert("RGB")

                img.save(buffer, format="JPEG")
                encoded_string = base64.b64encode(buffer.getvalue()).decode("utf-8")
                return f"data:image/jpeg;base64,{encoded_string}"
        except Exception as e:
            logger.error(f"Failed to encode image {image_path} to base64: {e}")
            raise

    def _api_call(self, system_prompt: str, user_input: str, temperature: float, max_tokens: int, img_path: str = None):
        client = openai.OpenAI(api_key=self.api_key, timeout=600)

        assert "{}" in system_prompt, "system_prompt must contain {{}}"
        prompt_text = system_prompt.format(user_input)

        assert img_path is not None, "img_path is required"
        base64_image = self._encode_image_to_base64(img_path, max_dimension=self.max_image_size)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": base64_image}},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

        last_err = None
        for i in range(5):
            try:
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return response.choices[0].message.content or ""
            except Exception as e:
                last_err = e
                logger.error(f"OpenAIVisionClient request failed (attempt {i + 1}/5): {e}")
                if i < 4:
                    time.sleep(2 ** i)
                else:
                    raise last_err

    def run_single_recaption(self, system_prompt, input_prompt, temperature=0.1, max_tokens=4096, img_path: str = None):
        return self._api_call(system_prompt, input_prompt, temperature, max_tokens, img_path=img_path)


class AzureOpenAIClient(object):
    """Client for Azure OpenAI API (text-only prompt rewriting)."""

    def __init__(self, model_name=None):
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01")
        self.deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        if not self.api_key or not self.endpoint:
            raise EnvironmentError(
                "AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT must be set in environment variables."
            )
        if not self.deployment_name:
            raise EnvironmentError(
                "AZURE_OPENAI_DEPLOYMENT_NAME must be set in environment variables."
            )
        self.model_name = self.deployment_name

    def _api_call(self, system_prompt: str, user_input: str, temperature: float, max_tokens: int):
        client = openai.AzureOpenAI(
            api_key=self.api_key,
            azure_endpoint=self.endpoint,
            api_version=self.api_version,
            timeout=600,
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ]

        last_err = None
        for i in range(5):
            try:
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return response.choices[0].message.content or ""
            except Exception as e:
                last_err = e
                if i < 4:
                    time.sleep(2 ** i)
                    continue
                raise last_err

    def run_single_recaption(self, system_prompt, input_prompt, temperature=0.1, max_tokens=4096):
        return self._api_call(system_prompt, input_prompt, temperature, max_tokens)


class AzureOpenAIVisionClient(object):
    """Client for Azure OpenAI API with vision support (image-to-video prompt rewriting)."""

    def __init__(self, model_name=None):
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01")
        self.deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        if not self.api_key or not self.endpoint:
            raise EnvironmentError(
                "AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT must be set in environment variables."
            )
        if not self.deployment_name:
            raise EnvironmentError(
                "AZURE_OPENAI_DEPLOYMENT_NAME must be set in environment variables."
            )
        self.model_name = self.deployment_name
        self.max_image_size = int(os.getenv("I2V_REWRITE_MAX_IMAGE_SIZE", "1024"))

    def _encode_image_to_base64(self, image_path: str, max_dimension: int) -> str:
        try:
            from PIL import Image
        except ImportError as e:
            logger.error("Pillow (PIL) is required for image encoding but is not installed.")
            raise e

        try:
            image = image_path
            if not isinstance(image, Image.Image):
                image = Image.open(image)

            with image as img:
                if img.width > max_dimension or img.height > max_dimension:
                    img.thumbnail((max_dimension, max_dimension))

                buffer = io.BytesIO()
                if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
                    img = img.convert("RGB")

                img.save(buffer, format="JPEG")
                encoded_string = base64.b64encode(buffer.getvalue()).decode("utf-8")
                return f"data:image/jpeg;base64,{encoded_string}"
        except Exception as e:
            logger.error(f"Failed to encode image {image_path} to base64: {e}")
            raise

    def _api_call(self, system_prompt: str, user_input: str, temperature: float, max_tokens: int, img_path: str = None):
        client = openai.AzureOpenAI(
            api_key=self.api_key,
            azure_endpoint=self.endpoint,
            api_version=self.api_version,
            timeout=600,
        )

        assert "{}" in system_prompt, "system_prompt must contain {{}}"
        prompt_text = system_prompt.format(user_input)

        assert img_path is not None, "img_path is required"
        base64_image = self._encode_image_to_base64(img_path, max_dimension=self.max_image_size)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": base64_image}},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

        last_err = None
        for i in range(5):
            try:
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return response.choices[0].message.content or ""
            except Exception as e:
                last_err = e
                logger.error(f"AzureOpenAIVisionClient request failed (attempt {i + 1}/5): {e}")
                if i < 4:
                    time.sleep(2 ** i)
                else:
                    raise last_err

    def run_single_recaption(self, system_prompt, input_prompt, temperature=0.1, max_tokens=4096, img_path: str = None):
        return self._api_call(system_prompt, input_prompt, temperature, max_tokens, img_path=img_path)


def create_text_rewrite_client(model_name=None):
    """Factory function to create the appropriate text rewrite client based on REWRITE_PROVIDER env var."""
    provider = _get_rewrite_provider()
    if provider == "openai":
        return OpenAIClient(model_name=model_name)
    elif provider == "azure":
        return AzureOpenAIClient(model_name=model_name)
    else:
        raise ValueError(f"Unsupported REWRITE_PROVIDER: {provider}. Must be 'openai' or 'azure'.")


def create_vision_rewrite_client(model_name=None):
    """Factory function to create the appropriate vision rewrite client based on REWRITE_PROVIDER env var."""
    provider = _get_rewrite_provider()
    if provider == "openai":
        return OpenAIVisionClient(model_name=model_name)
    elif provider == "azure":
        return AzureOpenAIVisionClient(model_name=model_name)
    else:
        raise ValueError(f"Unsupported REWRITE_PROVIDER: {provider}. Must be 'openai' or 'azure'.")