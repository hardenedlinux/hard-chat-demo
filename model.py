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

from typing import Any, Iterator
from llama_cpp import Llama
from sentencepiece import SentencePieceProcessor

def get_prompt(message: str,
               chat_history: list[tuple[str, str]],
               system_prompt: str) -> str:
    """Create a prompt for the model to generate a response from."""
    texts = [f"<|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>\n\n"]
    texts.append(f"<|start_header_id|>user<|end_header_id|>\n\n{message.strip()}<|eot_id|>\n\n")
    return "".join(texts)

class LLAMA_wrapper:
    """Wrapper for llama model."""

    def __init__(self, config: dict = {}):
        """Initialize the model."""
        self.config = config
        self.model = Llama(model_path=config["model_name"],
                           n_ctx=config["max_input"],
                           n_batch=config["max_input"])
        self.tokenizer = SentencePieceProcessor(model_file="tokenizer.model")

    def get_input_token_length(self,
                               message: str,
                               chat_history: list[tuple[str, str]],
                               system_prompt: str) -> int:
        """Get the input token length for a prompt."""
        prompt = get_prompt(message, chat_history, system_prompt)
        input_ids = self.tokenizer.EncodeAsIds(prompt)
        return len(input_ids)

    def generate(self,
                 prompt: str,
                 max_new_tokens: int = 1024,
                 temperature: float = 0.8,
                 top_p: float = 0.95,
                 top_k: int = 50,
                 ) -> Iterator[str]:
        """Generate a response from a prompt."""
        inputs = self.model.tokenize(bytes(prompt, "utf-8"))
        generate_kwargs = dict(top_p=top_p, top_k=top_k, temp=temperature)
        generator = self.model.generate(inputs, **generate_kwargs)
        outputs = []
        for token in generator:
            if token == self.model.token_eos():
                break
            b_text = self.model.detokenize([token])
            text = str(b_text, encoding="utf-8")
            outputs.append(text)
            yield "".join(outputs)

    def run(self,
            message: str,
            chat_history: list[tuple[str, str]],
            system_prompt: str,
            max_new_tokens: int = 1024,
            temperature: float = 0.8,
            top_p: float = 0.95,
            top_k: int = 50,
            ) -> Iterator[str]:
        """Generate a response from a prompt."""
        prompt = get_prompt(message, chat_history, system_prompt)
        return self.generate(prompt, max_new_tokens, temperature, top_p, top_k)

    def __call__(self, prompt: str, **kwargs: Any) -> str:
        """Generate a response from a prompt."""
        return self.model.__call__(prompt, **kwargs)["choices"][0]["text"]
