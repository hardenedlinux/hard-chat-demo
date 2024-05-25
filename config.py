"""-*-  indent-tabs-mode:nil; coding: utf-8 -*-.

Copyright (C) 2024
    HardenedLinux community
This is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.
If not, see <http://www.gnu.org/licenses/>.

"""

import os


def get_prompt() -> str:
    return os.getenv("CHAT_PROMPT") or """
    You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""


def get_model_path() -> str:
    mp = os.getenv("CHAT_MODEL_PATH")
    return mp or "models/llama-3-8B-q4-k-m.gguf"


def get_max_input() -> int:
    mi = os.getenv("CHAT_MAX_INPUT")
    return int(mi) if mi else 4000


def get_max_new_tokens() -> int:
    mnt = os.getenv("CHAT_MAX_NEW_TOKENS")
    return int(mnt) if mnt else 2048


config = {
    "model_name": get_model_path(),
    "max_input": get_max_input(),
    "prompt": get_prompt(),
    "max_new_tokens": get_max_new_tokens(),
}
