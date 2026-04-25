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

import os
from hyvideo.utils.rewrite.clients import create_text_rewrite_client, create_vision_rewrite_client
from hyvideo.utils.rewrite.t2v_prompt import t2v_rewrite_system_prompt
from hyvideo.utils.rewrite.i2v_prompt import i2v_rewrite_system_prompt

def t2v_rewrite(user_prompt, rewrite_client=None):
    model_name = os.getenv("REWRITE_MODEL_NAME")
    if rewrite_client is None:
        rewrite_client = create_text_rewrite_client(model_name=model_name)
    try:
        rewritten_prompt = rewrite_client.run_single_recaption(t2v_rewrite_system_prompt, user_prompt)
    except Exception as e:
        raise ValueError(
            f"Failed to rewrite prompt using {type(rewrite_client).__name__}: {e}"
        )
    return rewritten_prompt

def i2v_rewrite(user_input, img_path, rewrite_client=None):
    """
    Use a rewrite client to generate a rewritten prompt for image-to-video.
    """
    model_name = os.getenv("REWRITE_MODEL_NAME")
    if rewrite_client is None:
        rewrite_client = create_vision_rewrite_client(model_name=model_name)
    try:
        rewritten_prompt = rewrite_client.run_single_recaption(i2v_rewrite_system_prompt, user_input, img_path=img_path)
    except Exception as e:
        raise ValueError(
            f"Failed to rewrite prompt using {type(rewrite_client).__name__}: {e}"
        )
    return rewritten_prompt

def run_prompt_rewrite(user_prompt, img_path, task_type):
    if task_type == "i2v":
        return i2v_rewrite(user_prompt, img_path)
    elif task_type == "t2v":
        return t2v_rewrite(user_prompt)
    else:
        raise ValueError(f"Unsupported task_type: {task_type}. Must be 'i2v' or 't2v'")
